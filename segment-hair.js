/**
 * segment-hair.js
 *
 * Node.js 래퍼: Python GroundingDINO + SAM 기반 hair segmentation 호출
 *
 * 사용법:
 *   node segment-hair.js <image_path> [output_dir]
 *
 * 출력:
 *   <output_dir>/hair_mask.png
 *   <output_dir>/hair_mask_inv.png
 *
 * Python 의존성 사전 설치:
 *   pip install -r python/requirements.txt
 */

import { spawnSync }    from 'child_process';
import path             from 'path';
import fs               from 'fs';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ── 설정 ─────────────────────────────────────────────────────────────────────

const PYTHON_SCRIPT = path.join(__dirname, 'python', 'segment_hair.py');
const PYTHON_CMD    = process.platform === 'win32' ? 'python' : 'python3';

// ── CLI 인자 ──────────────────────────────────────────────────────────────────

const IMAGE_PATH = process.argv[2];
const OUTPUT_DIR = process.argv[3] || path.dirname(IMAGE_PATH) || '.';

if (!IMAGE_PATH) {
  console.error('Usage: node segment-hair.js <image_path> [output_dir]');
  console.error('  예: node segment-hair.js photo.png output/masks');
  process.exit(1);
}
if (!fs.existsSync(IMAGE_PATH)) {
  console.error(`[ERROR] 이미지 파일 없음: ${IMAGE_PATH}`);
  process.exit(1);
}
if (!fs.existsSync(PYTHON_SCRIPT)) {
  console.error(`[ERROR] Python 스크립트 없음: ${PYTHON_SCRIPT}`);
  process.exit(1);
}

// ── Python 환경 확인 ──────────────────────────────────────────────────────────

function checkPython() {
  const result = spawnSync(PYTHON_CMD, ['--version'], { encoding: 'utf8' });
  if (result.error) {
    console.error(`[ERROR] Python 실행 불가: ${result.error.message}`);
    console.error('  Python 3.9+ 이 PATH에 있어야 합니다.');
    process.exit(1);
  }
  console.log(`[INFO] ${result.stdout.trim() || result.stderr.trim()}`);
}

function checkDependencies() {
  const pkgs  = ['torch', 'transformers', 'segment_anything', 'cv2', 'PIL', 'numpy'];
  const check = spawnSync(PYTHON_CMD, ['-c', pkgs.map(p => `import ${p}`).join(';')], { encoding: 'utf8' });
  if (check.status !== 0) {
    console.warn('[WARN] 일부 Python 패키지가 없습니다. 설치 필요:');
    console.warn('  pip install -r python/requirements.txt');
    console.warn('  # segment-anything 별도 설치:');
    console.warn('  pip install segment-anything');
    console.warn(check.stderr.slice(0, 400));
    // 경고만 출력하고 계속 진행 (Python 스크립트에서 더 자세한 오류 출력됨)
  }
}

// ── 메인 ─────────────────────────────────────────────────────────────────────

function main() {
  console.log('='.repeat(55));
  console.log('  Hair Segmentation: GroundingDINO + SAM');
  console.log('='.repeat(55));
  console.log(`  이미지   : ${IMAGE_PATH}`);
  console.log(`  출력 폴더: ${OUTPUT_DIR}`);
  console.log('='.repeat(55));

  checkPython();
  checkDependencies();

  // 출력 디렉토리 생성
  if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });

  console.log('\n[RUN] Python segmentation 시작...\n');

  const args = [
    PYTHON_SCRIPT,
    path.resolve(IMAGE_PATH),
    '--output-dir', path.resolve(OUTPUT_DIR),
  ];

  const result = spawnSync(PYTHON_CMD, args, {
    encoding: 'utf8',
    stdio:    'inherit',      // stdout/stderr 실시간 출력
    cwd:      __dirname,
  });

  if (result.error) {
    console.error(`\n[ERROR] 프로세스 실행 오류: ${result.error.message}`);
    process.exit(1);
  }

  if (result.status !== 0) {
    console.error(`\n[ERROR] Python 스크립트 종료 코드: ${result.status}`);
    process.exit(result.status);
  }

  // 결과 파일 확인
  const hairMask    = path.join(OUTPUT_DIR, 'hair_mask.png');
  const hairMaskInv = path.join(OUTPUT_DIR, 'hair_mask_inv.png');

  console.log('\n' + '='.repeat(55));
  if (fs.existsSync(hairMask) && fs.existsSync(hairMaskInv)) {
    const sz1 = Math.round(fs.statSync(hairMask).size    / 1024);
    const sz2 = Math.round(fs.statSync(hairMaskInv).size / 1024);
    console.log('  완료!');
    console.log(`  hair_mask.png     → ${hairMask} (${sz1} KB)`);
    console.log(`  hair_mask_inv.png → ${hairMaskInv} (${sz2} KB)`);
  } else {
    console.error('  [ERROR] 출력 파일이 생성되지 않았습니다.');
    process.exit(1);
  }
  console.log('='.repeat(55));
}

main();
