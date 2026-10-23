# Kirikae

Recompilador estático de executáveis do Nintendo Switch para programas nativos de PC, em Rust.

```
kirikae build jogo.dnsp -o saida/    # ARM64 → IR → Cranelift → saida/jogo.exe
```

> Use apenas jogos extraídos do **seu próprio console**. O Kirikae não inclui, não baixa e não
> descriptografa nenhum arquivo de jogo: ele lê só dumps já descriptografados, o código é traduzido e
> compilado na máquina de quem joga, e o executável gerado não deve ser compartilhado. Regras
> completas em [LEGAL.md](LEGAL.md). O Kirikae não é afiliado à Nintendo.

## Diferenciais

- **Rust do começo ao fim.** O runtime (kernel, serviços, GPU, áudio e vídeo) é a parte maior e mais
  sujeita a bugs de memória, e é justamente onde Rust ajuda mais.
- **Uma única fonte de verdade para a CPU.** Cada instrução ARM64 é traduzida uma vez para um IR
  pequeno ([`crates/a64/src/ir.rs`](crates/a64/src/ir.rs)), e toda operação é uma função em
  [`ops.rs`](crates/a64/src/ops.rs). O interpretador e o backend Rust usam essas funções diretamente;
  o backend Cranelift traduz as operações por conta própria e é testado contra elas em milhares de
  entradas aleatórias.
- **Código gerado sem `unsafe`.** O espaço de endereços do Switch (39 bits) é reservado inteiro, e
  todo acesso passa por uma máscara: um ponteiro errado do jogo gera falha de página, nunca corrompe
  a memória do host.
- **Fallback, não crash.** Código que a análise estática não encontrou (jump tables, vtables) é
  interpretado em tempo de execução. O executável anota onde o interpretador começou
  (`<nome>.profile`), e o próximo `kirikae build` compila esses endereços também.
- **Licença permissiva** (MIT ou Apache-2.0), sem código GPL. As referências são o Switchbrew, o
  Ryujinx (MIT), o envytools (MIT), a documentação aberta da NVIDIA e o Arm ARM.
- **Sem dependências além do Rust.** O Cranelift gera o código de máquina em paralelo; o Cargo e o
  linker do sistema montam o `.exe`.

## Como funciona

```
 .dnsp / ExeFS / .nso / .nro ──► loader ──► módulos (rtld, main, sdk, subsdk*) nos endereços do guest
                                  │
                                  ▼
                             discover ── entradas: início de cada módulo, relocações e símbolos
                                  │      dinâmicos, alvos de BL, ADR, ADRP+ADD
                                  ▼
                     lift (a64) ── IR por instrução ◄── o mesmo IR que o interpretador usa
                                  │
                ┌─────────────────┴─────────────────┐
                ▼                                   ▼
   clif (padrão): Cranelift → .obj      rust_backend: Rust → rustc
   em paralelo, 1024 funções por .obj   (referência, programas pequenos)
                └─────────────────┬─────────────────┘
                                  ▼
          cargo ── jogo.exe = código traduzido + kirikae-rt + executável original embutido
```

Em tempo de execução, `kirikae-rt` reserva a memória do guest, mapeia os módulos, a pilha e o TLS e
chama a entrada. Chamadas diretas viram chamadas nativas; as indiretas passam por um dispatcher que
usa a função recompilada quando ela existe e o interpretador quando não existe. O runtime implementa o
kernel do Horizon (threads, sincronização, memória), os serviços do sistema, a GPU (o processador de
comandos do Maxwell sobre Vulkan, com os shaders traduzidos para SPIR-V), áudio e os decodificadores
de vídeo.

## Requisitos

- Windows 10/11 x64 e uma GPU com Vulkan 1.3.
- [Rust](https://rustup.rs) (o `cargo`) e o linker do MSVC (Visual Studio Build Tools com
  "Desenvolvimento para desktop com C++"): o jogo é compilado na sua máquina. A primeira compilação
  baixa as dependências do runtime pelo Cargo.
- Um dump **descriptografado** de um jogo seu: um `.dnsp` (PFS0 com a NCA de programa em texto claro),
  uma pasta ExeFS extraída (com `romfs.bin` ao lado, se houver), um `.nso` ou um `.nro`.

## Uso

### Interface gráfica

`kirikae-gui.exe`: escolha (ou arraste) os arquivos do jogo, a pasta de saída, e clique em
**Recompile**. O progresso aparece no log, e cada jogo pronto ganha um botão para abrir a pasta. A GUI
chama o `kirikae.exe` que estiver ao lado dela.

### Linha de comando

```bash
cargo build --release -p kirikae -p kirikae-gui
kirikae testgen hello.nro            # programa de teste embutido
kirikae run hello.nro                # só interpretador
kirikae build hello.nro -o out       # recompila para out/hello.exe (--backend rust para o backend Rust)
kirikae build jogo.dnsp -o out       # um jogo inteiro (DNSP ou pasta ExeFS)
kirikae build jogo.dnsp -o out --profile run.profile   # compila também o que uma execução interpretou
kirikae build jogo.dnsp -o out --plugin minha-crate    # liga uma crate cujo install() roda antes do jogo
KIRIKAE_STATS=1 out/jogo.exe         # fps, trabalho da GPU, chamadas nativas x interpretadas
kirikae info jogo.dnsp               # módulos, segmentos e cobertura do decodificador
kirikae dev syms jogo.dnsp nvn       # funções exportadas cujo nome contém "nvn"
kirikae dev module-dump jogo.dnsp sdk sdk.bin   # imagem carregada de um módulo
```

**Plugins.** `--plugin <pasta de uma crate>` compila a crate junto com o jogo e chama
`<crate>::install()` antes dele começar. Com `kirikae_rt::games::install` um plugin pode ajustar o
ritmo de apresentação e agir a cada quadro apresentado (por exemplo, corrigir o relógio de um jogo
que avança por quadro).

**Cobertura guiada por perfil.** O executável gerado anota cada endereço em que o interpretador
começou a rodar e grava a lista em `<nome>.profile` a cada 10 s (`KIRIKAE_PROFILE=<arquivo>` muda o
lugar, `0` desliga). O `kirikae build` seguinte para a mesma pasta lê esse arquivo (ou o de
`--profile`) e compila esses endereços como funções. Use o perfil só com o mesmo jogo e a mesma versão.

**Cobertura do que o jogo usa.** O executável registra cada comando IPC, SVC, ioctl do `nvdrv` e
instrução de shader usada, com o status (`implemented`, `stub`, `missing`), em `<nome>.coverage.json`,
e guarda um evento (backtrace, quadro em PNG, cópia dos saves) na primeira vez que algo não
implementado aparece. Depois de jogar: `python tools/coverage_report.py saida/jogo.coverage.json`.
Esses arquivos têm imagens e saves do jogo e ficam fora do git.

### Pacote de release

`python tools/package_release.py` compila o `kirikae.exe` e o `kirikae-gui.exe` e monta
`dist/kirikae-<versão>-windows-x64.zip` com o código do runtime em `runtime/` (os jogos são compilados
contra ele; `KIRIKAE_RUNTIME=<pasta de kirikae-rt>` aponta para outro).

Variáveis de ambiente úteis:

| Variável | Efeito |
| --- | --- |
| `KIRIKAE_PARALLEL=0` | desliga o paralelismo (ligado por padrão, por ser o mais rápido): volta ao lock global, com uma thread do guest por vez: útil para isolar corridas |
| `KIRIKAE_SYNC_PIPELINES=1` | compila os pipelines gráficos na hora do draw, como antes, em vez de em threads à parte (sem ele, um draw cujo pipeline ainda está compilando é pulado até ficar pronto, exceto em alvos pequenos, até 512x512, que não foram desenhados no frame anterior: esses passes raros fazem dados uma vez só e compilam na hora) |
| `KIRIKAE_STATS=1` | a cada 10 s: frames, fps (o que se alcançaria sem o limite de quadros, e o frame mais longo, onde as travadas aparecem; com `KIRIKAE_STATS=2`, cada quadro acima de 100 ms com o horário, para casar com um perfil por amostragem), o tempo de CPU de cada draw por etapa (alvos, shaders, pipeline, descritores, vértices, gravação...; só com `KIRIKAE_STATS=2`, que lê o relógio ~20 vezes por draw), trabalho da GPU, chamadas nativas e instruções interpretadas, e onde o tempo das threads do jogo foi por quadro (código nativo, interpretador, SVCs, serviços, GPU, espera do lock global ou do núcleo, espera do lock de kernel, bloqueado), as trocas do lock, o custo de cada SVC e as threads mais ocupadas |
| `KIRIKAE_PROFILE=<arquivo>` | onde gravar os endereços que o interpretador rodou (padrão `<exe>.profile`; `0` desliga), lidos pelo próximo `kirikae build` |
| `KIRIKAE_COVERAGE=<arquivo>` | onde gravar o que o jogo usou (padrão `<exe>.coverage.json`, eventos em `<exe>-coverage/`; `0` desliga); `KIRIKAE_COVERAGE_EVENTS=<n>` muda o limite de 200 eventos por execução |
| `KIRIKAE_SIMD_HIST=<arquivo>` | com `KIRIKAE_STATS`, grava quantas vezes cada instrução SIMD/FP passou pelo helper do runtime (e, em `<arquivo>.ops`, cada op que passou pelo `kirikae_op`) |
| `KIRIKAE_HEADLESS=1` | não abre a janela (os frames continuam sendo contados) |
| `KIRIKAE_FRAME_DUMP=<pasta>` | grava um a cada 60 frames apresentados como PNG (só esses frames são lidos de volta da GPU); `KIRIKAE_FRAME_DUMP_EVERY=<n>`, `KIRIKAE_FRAME_DUMP_FROM=<frame>` e `KIRIKAE_FRAME_DUMP_COUNT=<n>` mudam o intervalo, o início e quantos (frames seguidos mostram uma piscada) |
| `KIRIKAE_PRESENT_DUMP=<pasta>` | grava uma a cada 60 imagens da swapchain como PNG, do jeito que vão para a janela |
| `KIRIKAE_VSYNC=1` | apresenta esperando o vertical blank (FIFO); sem ela, MAILBOX ou IMMEDIATE |
| `KIRIKAE_GPU_TIME=frame` ou `live` | o tempo que os reports de timestamp da GPU escrevem: um por frame (o do último apresentado) ou o do momento (padrão) |
| `KIRIKAE_NO_FRAME_LIMIT=1` | não segura o ritmo de apresentação: ignora o intervalo de troca (vblanks por frame) que o jogo pede e deixa passar de 60 frames por segundo (jogos cujo mundo avança por quadro ficam acelerados) |
| `KIRIKAE_KEEP_NOOP_DRAWS=1` | mantém os draws que não escrevem nada (cor toda mascarada ou sem fragment shader, sem escrita de profundidade, sem stencil, shaders sem escrita em memória), que por padrão são pulados |
| `KIRIKAE_NVN=trace` | com `KIRIKAE_STATS`, conta as chamadas de cada função NVN do jogo (as mais chamadas por quadro a cada 10 s, e todas em `<exe>.nvn.txt`) |
| `KIRIKAE_NVN=shadow` | o mesmo, e monta no host o modelo do que o jogo pede ao NVN, comparado draw a draw com o que o processador de comandos decodifica (draws, estado, clears e os endereços de pools, buffers, texturas e views que o modelo calcula; contagens com `KIRIKAE_STATS`, as primeiras diferenças no log; bem mais lento) |
| `KIRIKAE_NO_CHECKED=1` | confere de novo cada constant buffer e vertex stream a cada draw, em vez de reaproveitar o que já foi conferido no mesmo lote (para isolar erros) |
| `KIRIKAE_NO_BUFFER_CACHE=1` | copia vertex, index e constant buffers a cada draw, sem o cache na GPU |
| `KIRIKAE_SAVES=<pasta>` | a pasta dos saves; cada jogo usa `<pasta>\<title id>` (padrão `%APPDATA%\kirikae\saves`) |
| `KIRIKAE_SHARED_FONT=<arquivo.ttf>` | fonte usada no lugar das fontes do sistema |
| `KIRIKAE_MUTE=1` | não abre o dispositivo de som (o áudio continua sendo mixado, então o jogo segue normal) |
| `KIRIKAE_AUDIO_DUMP=<arquivo.wav>` | grava tudo o que o renderizador de áudio mixa num WAV estéreo de 16 bits |
| `KIRIKAE_TRACE_AUDIO=1` | registra vozes, mixes, sink e efeitos do `audren` e o estado das vozes a cada 2 s |
| `KIRIKAE_TRACE_VIDEO=1` | registra cada quadro que o NVDEC decodifica (codec, tamanho, tempo) e cada composição do VIC |
| `KIRIKAE_VIDEO_DUMP=<pasta>` | grava os quadros H.264 que o NVDEC recebe (com SPS/PPS reconstruídos) num `.h264` tocável e cada 100º quadro VP9 decodificado como NV12 cru |
| `KIRIKAE_TRACE_IPC=1`, `KIRIKAE_TRACE_SVC=1`, `KIRIKAE_TRACE_GPU=1` | registram serviços, SVCs e comandos da GPU (`KIRIKAE_TRACE_GPU_FROM=<frame>` começa a registrar a GPU só a partir desse frame, `KIRIKAE_TRACE_GPU_COUNT=<n>` por tantos frames) |
| `KIRIKAE_STRICT_IPC=1` | comandos não implementados falham em vez de responder zeros |
| `KIRIKAE_NO_VULKAN=1` | não renderiza (só a apresentação por CPU dos buffers da janela) |
| `KIRIKAE_NPAD=handheld` | conecta o controle portátil em vez do controle Pro no jogador 1 |
| `KIRIKAE_INPUT=<roteiro>` | aperta botões por tempo, para testes sem janela: `40:A,45-47:LUP+B` (segundos desde o primeiro uso do `hid`; um tempo só segura 0,2 s; botões `A B X Y L R ZL ZR PLUS MINUS UP DOWN LEFT RIGHT LSTICK RSTICK`, direções de analógico `LUP`, `RLEFT`, ...) |

Controles: o jogo vê um controle Pro no jogador 1, movido pelo teclado (com a janela em foco) e pelo
primeiro gamepad XInput (Xbox e compatíveis, botões pela posição: o de baixo é B, o da direita é A).
As teclas valem pela posição num teclado americano, então não mudam em ABNT2:

| Switch | Teclas |
| --- | --- |
| analógico esquerdo | WASD ou setas |
| analógico direito | IJKL |
| A, B, X, Y | Z ou Espaço, X, C, V |
| L, R, ZL, ZR | E, U, Q, O |
| +, − | Enter ou `=`, Backspace ou `-` (e + e − do teclado numérico) |
| direcional | T, F, G, H (cima, esquerda, baixo, direita) |
| clique dos analógicos | Shift esquerdo (esquerdo), M (direito) |

Para depurar os gráficos (lentos; servem para achar o passo que dá errado):

| Variável | Efeito |
| --- | --- |
| `KIRIKAE_TARGET_DUMP=<pasta>` | grava os render targets escritos pela GPU como PNG (a cada `KIRIKAE_TARGET_DUMP_EVERY` frames, padrão 60; `KIRIKAE_TARGET_DUMP_ONLY=<endereço>[,<endereço>...]` filtra; `KIRIKAE_TARGET_DUMP_RAW=1` grava também os bytes crus, inclusive de formatos que não viram imagem, como D16) |
| `KIRIKAE_TEX_SOLID=<endereço GPU>:<byte>[,...]` | as texturas nesses endereços passam a ler esse valor em todos os canais (para achar qual textura controla um efeito) |
| `KIRIKAE_TRACE_REGS=1` | com `KIRIKAE_TRACE_GPU`, imprime todos os registradores 3D não nulos de cada draw (para comparar passes) |
| `KIRIKAE_TEX_DUMP=<pasta>:<endereço>[+<endereço>...]` | grava os texels (lineares, todos os níveis e camadas) de cada envio da textura nesse endereço (da CPU ou da GPU) |
| `KIRIKAE_SHADER_DUMP=<pasta>`, `KIRIKAE_SPIRV_DUMP=<pasta>` | grava os shaders Maxwell usados e o SPIR-V gerado |
| `KIRIKAE_CBUF_DUMP=<pasta>:<endereço do shader>[+<endereço>...]`, `KIRIKAE_SSBO_DUMP=<pasta>` | grava os constant buffers que um shader lê (um arquivo por draw; `KIRIKAE_CBUF_DUMP_FRAME=<n>` limita a um frame) e os storage buffers que a GPU escreve |
| `KIRIKAE_PROBE=[shader:][pc:]rN` | o shader escreve os registradores N..N+2 (no `pc` dado ou no fim) no lugar da cor do alvo 0 e N+3..N+5 no alvo 1 (`KIRIKAE_PROBE_SIGNED=1`: o alvo 1 recebe N..N+2 negados, para ver sinais em alvos sem sinal); nos compute shaders vai para um buffer gravado em `KIRIKAE_PROBE_DUMP=<pasta>` |
| `KIRIKAE_SKIP_SHADER=<endereço>[,...]` | pula os draws desses shaders (para ver o que cada um desenha) |
| `KIRIKAE_TARGET_DUMP_DERIVED=1` | com `KIRIKAE_TARGET_DUMP`, grava também as cópias feitas de render targets (arrays juntados, crops); com `KIRIKAE_TARGET_DUMP_RAW` o `.raw` traz todas as camadas |
| `KIRIKAE_DEBUG_SOLID=1`, `KIRIKAE_WHITE_TEXTURES=1` | fragment shaders saem magenta; texturas viram branco |
| `KIRIKAE_NO_DEPTH_TEST=1`, `KIRIKAE_NO_CULL=1` | desligam o teste de profundidade e o culling |
| `KIRIKAE_DRAW_PARTIAL=1` | desenha mesmo com instruções de shader não traduzidas |
| `KIRIKAE_PIPELINE_CACHE=<arquivo>` | onde guardar o cache de pipelines (vazio: nenhum; o padrão fica em `%LOCALAPPDATA%\kirikae`) |

No Windows, se o projeto estiver num caminho muito longo, aponte o `CARGO_TARGET_DIR` para uma pasta
curta (o linker do MSVC falha com caminhos acima de 260 caracteres).

## Progresso

![Progresso: cada quadrado é um comando de serviço, SVC, ioctl ou instrução de shader](docs/progress/progress.svg)

`python tools/progress.py` (sem dependências) lê o código do runtime e conta o que cada serviço,
SVC e ioctl do `nvdrv` já trata, contra o que o [Switchbrew](https://switchbrew.org) documenta
(`tools/reference/horizon.txt`, só IDs e nomes), e quantos mnemônicos de shader do Maxwell o
tradutor cobre, contra a tabela de opcodes do envytools. Ele gera `docs/progress/`: a imagem acima, a
página [`index.html`](docs/progress/index.html) com a tabela por interface e a lista do que falta
em cada uma, o `progress.json` e um histórico com um ponto por dia. Os comandos que só devolvem
sucesso ou constantes aparecem separados (verde-claro). O total cobre as interfaces que o kirikae
expõe ou que os jogos pedem, não todo o sistema. Com `--coverage <arquivo ou pasta>` ele
marca também o que os jogos jogados usaram (ver "Cobertura do que o jogo usa" acima).

## Estado

Jogos comerciais 3D completos já rodam: menus, vídeos, áudio e gameplay, a 30 quadros por segundo em
uma GPU modesta. O que falta aparece no relatório de progresso acima e no `coverage_report.py` de cada
jogo. Em andamento: o NVN (a API gráfica do console) implementado no host sobre Vulkan, no lugar do
driver do jogo (`KIRIKAE_NVN=trace` e `shadow` são os primeiros passos), para cortar o custo do
processador de comandos.

## Licença

MIT ([LICENSE-MIT](LICENSE-MIT)) ou Apache-2.0 ([LICENSE-APACHE](LICENSE-APACHE)), à sua escolha.
O material de terceiros e as regras para contribuir estão em [LEGAL.md](LEGAL.md).
