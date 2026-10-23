use crate::Error;

fn byte(src: &[u8], i: &mut usize) -> Result<u8, Error> {
    let b = *src.get(*i).ok_or(Error::Malformed("truncated LZ4 block"))?;
    *i += 1;
    Ok(b)
}

fn length(src: &[u8], i: &mut usize, nibble: u8) -> Result<usize, Error> {
    let mut len = nibble as usize;
    if nibble == 15 {
        loop {
            let b = byte(src, i)?;
            len += b as usize;
            if b != 255 {
                break;
            }
        }
    }
    Ok(len)
}

pub fn decompress(src: &[u8], out_len: usize) -> Result<Vec<u8>, Error> {
    let mut out = Vec::with_capacity(out_len);
    let mut i = 0;
    while i < src.len() {
        let token = byte(src, &mut i)?;
        let literals = length(src, &mut i, token >> 4)?;
        let lit = src.get(i..i + literals).ok_or(Error::Malformed("truncated LZ4 literals"))?;
        out.extend_from_slice(lit);
        i += literals;
        if i == src.len() {
            break;
        }
        let offset = byte(src, &mut i)? as usize | (byte(src, &mut i)? as usize) << 8;
        if offset == 0 || offset > out.len() {
            return Err(Error::Malformed("bad LZ4 match offset"));
        }
        let matched = length(src, &mut i, token & 15)? + 4;
        let start = out.len() - offset;
        if offset >= matched {
            out.extend_from_within(start..start + matched);
        } else {
            for k in 0..matched {
                out.push(out[start + k]);
            }
        }
        if out.len() > out_len {
            return Err(Error::Malformed("LZ4 output larger than expected"));
        }
    }
    if out.len() != out_len {
        return Err(Error::Malformed("LZ4 output size mismatch"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literals_and_overlapping_match() {
        let block = [0x35, b'a', b'b', b'c', 3, 0, 0x10, b'!'];
        assert_eq!(decompress(&block, 13).unwrap(), b"abcabcabcabc!");
    }

    #[test]
    fn rejects_bad_offsets() {
        assert!(decompress(&[0x10, b'a', 5, 0], 10).is_err());
    }
}
