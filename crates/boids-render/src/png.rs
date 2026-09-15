//! PNG output for screenshots.
//!
//! # Why a hand-written encoder
//!
//! The only image this project ever writes is a screenshot, and the only consumer is a human opening a
//! file. A full PNG encoder would mean an image-codec dependency, which for one function is a poor
//! trade: dependency trees are the thing most likely to break a three-day build, and this is the kind
//! of code whose correctness can be checked byte by byte and then never touched again.
//!
//! What is implemented is the smallest *valid* PNG: no filtering, and `zlib` streams built entirely
//! from uncompressed "stored" deflate blocks. The files are therefore larger than a compressed PNG
//! (roughly 1.1x the raw RGBA pixels) and completely unambiguous. Every decider reads them, including
//! browsers, ImageMagick and `pngcheck`.
//!
//! # Conventions
//!
//! Input rows are top-to-bottom RGBA8, non-premultiplied, which is the layout `wgpu` gives back for an
//! `Rgba8UnormSrgb` target. Note that all three of those properties matter: a PNG decoder must be told
//! the colour type (6 = truecolour with alpha), the bit depth (8) and the row order (top first), and
//! every one of them has a "looks almost right" failure mode.

/// Writes an RGBA8 image as a PNG.
///
/// # Errors
/// Returns a message when the pixel buffer does not match `width * height * 4` bytes, or when the file
/// cannot be written.
pub fn write_png_rgba(path: &std::path::Path, width: u32, height: u32, pixels: &[u8]) -> Result<(), String> {
    let expected = width as usize * height as usize * 4;
    if pixels.len() != expected {
        return Err(format!(
            "pixel buffer is {} bytes but {width}x{height} RGBA needs {expected}",
            pixels.len()
        ));
    }
    let bytes = encode_png_rgba(width, height, pixels);
    std::fs::write(path, &bytes).map_err(|e| format!("writing {}: {e}", path.display()))
}

/// Encodes an RGBA8 image as PNG bytes.
#[must_use]
pub fn encode_png_rgba(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
    // Raw scanlines: each row is prefixed with a filter byte. Filter type 0 means "no filtering", which
    // is what keeps this encoder simple; the cost is file size, not correctness.
    let row_bytes = width as usize * 4;
    let mut raw = Vec::with_capacity((row_bytes + 1) * height as usize);
    for row in 0..height as usize {
        raw.push(0u8);
        let start = row * row_bytes;
        raw.extend_from_slice(&pixels[start..start + row_bytes]);
    }

    let mut out = Vec::with_capacity(raw.len() + 128);
    out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(6); // colour type: truecolour with alpha
    ihdr.push(0); // compression method: deflate
    ihdr.push(0); // filter method: adaptive, as declared by each scanline
    ihdr.push(0); // interlace: none
    write_chunk(&mut out, b"IHDR", &ihdr);
    write_chunk(&mut out, b"IDAT", &zlib_stored(&raw));
    write_chunk(&mut out, b"IEND", &[]);
    out
}

/// Builds a zlib stream whose deflate payload is a sequence of stored (uncompressed) blocks.
///
/// A stored block carries a 16-bit length, so the payload is split into chunks no larger than 65535
/// bytes. `BFINAL` is set on the last block; getting that flag wrong produces a file that decodes
/// partially, which is a much more annoying failure than a rejected one.
#[must_use]
pub fn zlib_stored(data: &[u8]) -> Vec<u8> {
    const MAX_BLOCK: usize = 65_535;
    let mut out = Vec::with_capacity(data.len() + data.len() / MAX_BLOCK * 5 + 16);
    // zlib header: deflate, 32 KiB window, no preset dictionary, and a check value that makes the
    // whole two-byte header a multiple of 31 as the format requires.
    out.push(0x78);
    out.push(0x01);

    if data.is_empty() {
        // An empty stored block is still required: a deflate stream must contain at least one block.
        out.push(0x01);
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(!0u16).to_le_bytes());
    } else {
        let mut offset = 0;
        while offset < data.len() {
            let len = MAX_BLOCK.min(data.len() - offset);
            let last = offset + len == data.len();
            out.push(if last { 0x01 } else { 0x00 });
            #[allow(clippy::cast_possible_truncation)]
            let len_u16 = len as u16;
            out.extend_from_slice(&len_u16.to_le_bytes());
            out.extend_from_slice(&(!len_u16).to_le_bytes());
            out.extend_from_slice(&data[offset..offset + len]);
            offset += len;
        }
    }

    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

/// Adler-32 checksum, the one zlib uses.
#[must_use]
pub fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65_521;
    let (mut a, mut b) = (1u32, 0u32);
    for byte in data {
        a = (a + u32::from(*byte)) % MOD;
        b = (b + a) % MOD;
    }
    (b << 16) | a
}

/// CRC-32 as PNG uses it (reflected, polynomial 0xEDB88320).
#[must_use]
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for byte in data {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Appends a PNG chunk: length, type, data, and the CRC over type and data.
fn write_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    #[allow(clippy::cast_possible_truncation)]
    let len = data.len() as u32;
    out.extend_from_slice(&len.to_be_bytes());
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_known_values() {
        // The canonical check value for CRC-32/ISO-HDLC.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn adler32_matches_known_values() {
        // From the zlib documentation.
        assert_eq!(adler32(b""), 1);
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }

    #[test]
    fn zlib_stream_is_decodable_by_construction() {
        // Header check value: the two header bytes interpreted big-endian must be a multiple of 31.
        let stream = zlib_stored(b"hello world");
        assert_eq!(u16::from_be_bytes([stream[0], stream[1]]) % 31, 0);
        // Single stored block, final, with a length of 11.
        assert_eq!(stream[2], 0x01);
        assert_eq!(u16::from_le_bytes([stream[3], stream[4]]), 11);
        assert_eq!(u16::from_le_bytes([stream[5], stream[6]]), !11u16);
        assert_eq!(&stream[7..18], b"hello world");
        // Trailing Adler-32 over the uncompressed data.
        assert_eq!(
            u32::from_be_bytes(stream[18..22].try_into().unwrap()),
            adler32(b"hello world")
        );
    }

    #[test]
    fn multi_block_stream_splits_and_marks_the_end() {
        let data = vec![7u8; 100_000];
        let stream = zlib_stored(&data);
        // 65535 + 34465: two blocks, only the second marked final.
        assert_eq!(stream[2], 0x00);
        assert_eq!(u16::from_le_bytes([stream[3], stream[4]]), 65_535);
        let second = 2 + 5 + 65_535;
        assert_eq!(stream[second], 0x01);
        assert_eq!(
            u16::from_le_bytes([stream[second + 1], stream[second + 2]]),
            34_465
        );
    }

    #[test]
    fn empty_input_still_produces_a_valid_block() {
        let stream = zlib_stored(b"");
        assert_eq!(stream[2], 0x01);
        assert_eq!(u16::from_le_bytes([stream[3], stream[4]]), 0);
    }

    #[test]
    fn encoded_png_has_the_expected_structure() {
        let pixels: Vec<u8> = (0..16 * 16 * 4).map(|i| (i % 251) as u8).collect();
        let png = encode_png_rgba(16, 16, &pixels);

        assert_eq!(&png[0..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        // IHDR: length 13, type IHDR, big-endian dimensions, then the five fixed bytes.
        assert_eq!(u32::from_be_bytes(png[8..12].try_into().unwrap()), 13);
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(u32::from_be_bytes(png[16..20].try_into().unwrap()), 16);
        assert_eq!(u32::from_be_bytes(png[20..24].try_into().unwrap()), 16);
        assert_eq!(&png[24..29], &[8, 6, 0, 0, 0]);
        // The IHDR CRC must verify.
        assert_eq!(
            u32::from_be_bytes(png[29..33].try_into().unwrap()),
            crc32(&png[12..29])
        );
        // IEND must be present and empty.
        assert_eq!(&png[png.len() - 8..png.len() - 4], b"IEND");

        // Every chunk in the file must have a valid CRC, which is what a decoder checks first.
        let mut offset = 8;
        let mut chunks = 0;
        while offset < png.len() {
            let len = u32::from_be_bytes(png[offset..offset + 4].try_into().unwrap()) as usize;
            let end = offset + 8 + len;
            let expected = u32::from_be_bytes(png[end..end + 4].try_into().unwrap());
            assert_eq!(crc32(&png[offset + 4..end]), expected, "chunk at {offset}");
            chunks += 1;
            offset = end + 4;
        }
        assert_eq!(chunks, 3, "expected IHDR, IDAT and IEND");
        assert_eq!(offset, png.len(), "trailing bytes after IEND");
    }

    #[test]
    fn size_mismatch_is_reported() {
        let err = write_png_rgba(std::path::Path::new("/dev/null"), 4, 4, &[0u8; 10])
            .expect_err("a short buffer must be rejected");
        assert!(err.contains("needs 64"), "unexpected message: {err}");
    }
}
