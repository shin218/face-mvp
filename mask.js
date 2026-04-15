const sharp = require("sharp");

async function createMasks(inputPath) {
  const image = sharp(inputPath);
  const metadata = await image.metadata();

  const width = metadata.width;
  const height = metadata.height;

  const hairHeight = Math.floor(height * 0.12);   // 🔥 핵심
  const faceHeight = Math.floor(height * 0.48);   // 살짝 늘림
  const clothesHeight = height - hairHeight - faceHeight;

  const createMask = async (top, h, output) => {
    const mask = Buffer.alloc(width * height, 0);

    for (let y = top; y < top + h; y++) {
      for (let x = 0; x < width; x++) {
        mask[y * width + x] = 255;
      }
    }

    await sharp(mask, {
      raw: { width, height, channels: 1 },
    })
      .png()
      .toFile(output);
  };

  const invertMask = async (input, output) => {
    await sharp(input)
      .negate()
      .png()
      .toFile(output);
  };

  await createMask(0, hairHeight, "hair_mask.png");
  await createMask(hairHeight, faceHeight, "face_mask.png");
  await createMask(hairHeight + faceHeight, clothesHeight, "clothes_mask.png");

  await invertMask("hair_mask.png", "hair_mask_inv.png");
  await invertMask("face_mask.png", "face_mask_inv.png");
  await invertMask("clothes_mask.png", "clothes_mask_inv.png");

  console.log("✅ mask + 반전 mask 생성 완료");
}

const input = process.argv[2];

if (!input) {
  console.log("Usage: node mask.js <image>");
  process.exit(1);
}

createMasks(input).catch((err) => {
  console.error("❌ 오류:", err.message);
  process.exit(1);
});