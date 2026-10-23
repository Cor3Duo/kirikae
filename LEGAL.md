# Aviso legal e regras do projeto

O Kirikae é um recompilador: ele lê um executável do Nintendo Switch que você já tem e gera um programa
equivalente para o PC. O projeto existe para preservação, estudo e para jogar no PC os jogos que
você comprou. Este documento diz o que o Kirikae é, o que ele não é, e as regras para quem usa e para
quem contribui.

*Isto não é aconselhamento jurídico. As leis mudam de país para país; na dúvida, consulte um
advogado.*

## O que o repositório contém

- Código próprio, sob MIT ou Apache-2.0 ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)).
- Nenhum jogo, firmware, chave, BIOS, fonte do sistema, SDK, shader ou arte da Nintendo ou de
  terceiros. Os testes usam programas sintéticos gerados pelo próprio Kirikae.
- Nenhum código de descriptografia: o Kirikae lê apenas dumps já descriptografados.
- Nenhum executável gerado a partir de um jogo.

## Regras para quem usa

1. **Use só jogos do seu próprio console.** Faça o dump do jogo no seu Switch. Arquivos baixados da
   internet, mesmo de um jogo que você comprou, não são um dump seu.
2. **Não distribua o que o Kirikae gera.** O `.exe` recompilado contém o código do jogo traduzido, e o
   mesmo vale para os símbolos de depuração (`.pdb`), o perfil (`.profile`), os saves e as capturas.
   Tudo isso fica na sua máquina.
3. **Não peça nem ofereça jogos, chaves ou firmware** nas issues, nos pull requests ou em qualquer
   canal do projeto. Esses pedidos são fechados sem resposta.

## Regras para quem contribui

1. **Engenharia reversa limpa.** As referências aceitas são documentação pública e código com licença
   compatível: o [Switchbrew](https://switchbrew.org), o Ryujinx (MIT), o envytools (MIT), a
   documentação aberta da NVIDIA (open-gpu-doc, MIT), o Arm Architecture Reference Manual e as
   especificações do Vulkan. O que vier de projetos MIT/BSD mantém o aviso de copyright original.
2. **Nada de código GPL** (yuzu, suyu, ScarletNX e derivados): ele obrigaria o projeto inteiro a
   virar GPL.
3. **Nada de material vazado ou sob NDA**: SDK da Nintendo, documentação de desenvolvedor, código-fonte
   de jogos.
4. **Nada de dados de jogos nos testes ou na documentação**: nada de binários, shaders extraídos,
   texturas, capturas de tela ou dumps de memória. Descrever o comportamento observado (um registrador,
   um campo de estrutura, a ordem de uma inicialização) é aceitável; copiar o conteúdo, não.
5. **Sem links** para sites de ROMs, chaves ou firmware, em lugar nenhum do repositório.

## Material de terceiros

- [`tools/reference/horizon.txt`](tools/reference/horizon.txt): números e nomes de comandos IPC,
  SVCs e ioctls documentados no [Switchbrew](https://switchbrew.org), sob CC BY-SA. Esse arquivo
  segue a licença do Switchbrew, não a do Kirikae, e é usado só pela ferramenta de progresso.
- [`crates/maxwell/src/opcodes.rs`](crates/maxwell/src/opcodes.rs): tabela de opcodes gerada a partir
  do `envydis/gm107.c` do envytools (MIT); o aviso original está no topo do arquivo.
- [`crates/rt/src/nvn/resources.rs`](crates/rt/src/nvn/resources.rs): os tamanhos de bloco por
  formato de textura do NVN seguem a tabela do Savage Wind Recomp (MIT, Copyright (c) 2026 Revivedx
  and the Savage Wind Recomp contributors; Copyright (c) 2026 Peiikai).
- Dependências do Cargo: todas sob licenças permissivas (MIT, Apache-2.0, BSD, Zlib, ISC, Unlicense).
  O decodificador H.264 usa o OpenH264 da Cisco compilado do código-fonte (BSD-2). A licença de
  patentes de H.264 que a Cisco oferece cobre apenas os binários distribuídos pela própria Cisco, e
  não um OpenH264 compilado por você; quem for distribuir binários do Kirikae deve levar isso em conta.

## Marcas

Nintendo Switch, Nintendo e NVN são marcas dos seus donos. O Kirikae não é
afiliado à Nintendo nem aprovado por ela, e os nomes aparecem só para identificar compatibilidade.
