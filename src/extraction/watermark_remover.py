"""
SynthID Watermark Remover — Signature-Based Approach

Uses watermark signatures extracted from pure black/white Gemini images
to perform targeted watermark subtraction, combined with JPEG compression
for maximum effectiveness.

Key findings from analysis:
- Pure black images reveal the exact watermark as pixel values > 0
- 24/25 black images share the same pattern (r=0.74), indicating a fixed key
- JPEG Q50 + Signature subtraction gives 15-19% phase reduction at 34-38dB PSNR
- The watermark is content-adaptive, but has a fixed structural component
"""

import os
import sys
import io
import json
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import zoom
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

# Ensure same-directory modules are importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class RemovalResult:
    """Result of watermark removal."""
    success: bool
    cleaned_image: np.ndarray
    psnr: float
    ssim: float
    detection_before: Optional[Dict] = None
    detection_after: Optional[Dict] = None
    method: str = ''
    details: Dict = field(default_factory=dict)


class WatermarkRemover:
    """
    SynthID watermark remover using extracted signatures.
    
    Approach:
    1. Load pre-extracted watermark signature from pure black/white Gemini images
    2. Resize signature to match target image
    3. Subtract signature from image (disrupts fixed watermark component)
    4. Apply JPEG compression (disrupts remaining adaptive component)
    """
    
    def __init__(
        self,
        signature_dir: str = None,
        extractor=None
    ):
        """
        Args:
            signature_dir: Path to directory containing signature .npy files
            extractor: RobustSynthIDExtractor instance for verification
        """
        self.extractor = extractor
        self.signature = None
        self.white_signature = None
        self.meta = None
        
        if signature_dir:
            self.load_signature(signature_dir)
    
    def load_signature(self, signature_dir: str):
        """Load watermark signature from pre-extracted files."""
        black_path = os.path.join(signature_dir, 'synthid_black_signature.npy')
        white_path = os.path.join(signature_dir, 'synthid_white_signature.npy')
        meta_path = os.path.join(signature_dir, 'signature_meta.json')
        
        if os.path.exists(black_path):
            self.signature = np.load(black_path)
        
        if os.path.exists(white_path):
            self.white_signature = np.load(white_path)
        
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)
    
    def extract_signature_from_images(
        self,
        black_dir: str = None,
        white_dir: str = None,
        output_dir: str = None
    ):
        """
        Extract watermark signature directly from pure black/white Gemini images.
        
        On a pure black image, every pixel > 0 IS the watermark.
        On a pure white image, every pixel < 255 IS the watermark.
        """
        import glob
        
        if black_dir:
            black_files = sorted(glob.glob(os.path.join(black_dir, '*.png')))
            print(f"Found {len(black_files)} black images")
            
            # Load all and cluster by correlation to find main group
            all_wms = []
            for f in black_files:
                img = cv2.imread(f)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                all_wms.append(img_rgb.astype(np.float32))
            
            # Simple clustering: find the majority group
            n = len(all_wms)
            if n > 2:
                # Check pairwise correlation of flattened binary masks
                binary_wms = [(wm > 0).astype(np.float32).ravel() for wm in all_wms]
                corr_matrix = np.zeros((n, n))
                for i in range(n):
                    for j in range(i+1, n):
                        c = np.corrcoef(binary_wms[i], binary_wms[j])[0, 1]
                        corr_matrix[i, j] = c
                        corr_matrix[j, i] = c
                
                # Find largest group with r > 0.5
                groups = []
                visited = set()
                for i in range(n):
                    if i in visited:
                        continue
                    group = [i]
                    for j in range(i+1, n):
                        if j not in visited and corr_matrix[i, j] > 0.5:
                            group.append(j)
                    for g in group:
                        visited.add(g)
                    groups.append(group)
                
                # Use the largest group
                main_group = max(groups, key=len)
                print(f"Main group: {len(main_group)} images (excluded {n - len(main_group)} outliers)")
            else:
                main_group = list(range(n))
            
            self.signature = np.mean([all_wms[i] for i in main_group], axis=0)
            print(f"Signature shape: {self.signature.shape}")
        
        if white_dir:
            white_files = sorted(glob.glob(os.path.join(white_dir, '*.png')))
            print(f"Found {len(white_files)} white images")
            
            white_wms = []
            for f in white_files:
                img = cv2.imread(f)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                white_wms.append(255.0 - img_rgb.astype(np.float32))
            
            self.white_signature = np.mean(white_wms, axis=0)
        
        # Save if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            if self.signature is not None:
                np.save(os.path.join(output_dir, 'synthid_black_signature.npy'), self.signature)
            if self.white_signature is not None:
                np.save(os.path.join(output_dir, 'synthid_white_signature.npy'), self.white_signature)
            
            meta = {
                'black_shape': list(self.signature.shape) if self.signature is not None else None,
                'white_shape': list(self.white_signature.shape) if self.white_signature is not None else None,
                'recommended_subtraction_scale': 1.0,
                'recommended_jpeg_quality': 50,
            }
            with open(os.path.join(output_dir, 'signature_meta.json'), 'w') as f:
                json.dump(meta, f, indent=2)
            
            print(f"Saved to {output_dir}")
    
    def _resize_signature(self, target_h: int, target_w: int) -> np.ndarray:
        """Resize signature to match target image dimensions."""
        if self.signature is None:
            raise ValueError("No signature loaded. Call load_signature() first.")
        
        sig_h, sig_w = self.signature.shape[:2]
        if sig_h == target_h and sig_w == target_w:
            return self.signature
        
        scale_y = target_h / sig_h
        scale_x = target_w / sig_w
        return zoom(self.signature, (scale_y, scale_x, 1), order=1)
    
    @staticmethod
    def _jpeg_compress(image: np.ndarray, quality: int = 50) -> np.ndarray:
        """Apply JPEG compression/decompression."""
        img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode='RGB')
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        return np.array(Image.open(buf)).astype(np.float32)
    
    @staticmethod
    def compute_psnr(original: np.ndarray, modified: np.ndarray) -> float:
        """Compute Peak Signal-to-Noise Ratio."""
        mse = np.mean((original.astype(float) - modified.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return float(10 * np.log10(255.0 ** 2 / mse))
    
    @staticmethod
    def compute_ssim(original: np.ndarray, modified: np.ndarray) -> float:
        """Compute simplified SSIM."""
        from scipy import ndimage
        C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        
        orig_f = original.astype(np.float64)
        mod_f = modified.astype(np.float64)
        
        mu1 = ndimage.uniform_filter(orig_f, size=11)
        mu2 = ndimage.uniform_filter(mod_f, size=11)
        
        sigma1_sq = ndimage.uniform_filter(orig_f ** 2, size=11) - mu1 ** 2
        sigma2_sq = ndimage.uniform_filter(mod_f ** 2, size=11) - mu2 ** 2
        sigma12 = ndimage.uniform_filter(orig_f * mod_f, size=11) - mu1 * mu2
        
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(np.mean(ssim_map))
    
    def remove(
        self,
        image: np.ndarray,
        mode: str = 'balanced',
        verify: bool = True,
        strength: str = 'aggressive'
    ) -> RemovalResult:
        """
        Remove SynthID watermark from image.
        
        Args:
            image: Input image (RGB, uint8)
            mode: 'light', 'balanced', 'aggressive', 'maximum', or 'combined_worst'
            verify: Whether to verify removal with detection
            
        Returns:
            RemovalResult with cleaned image and metrics
        """
        # V2 combined worst-case mode — delegates to bypass_v2 pipeline
        if mode == 'combined_worst':
            return self._remove_combined_worst(image, verify=verify, strength=strength)
        
        img_f = image.astype(np.float32)
        h, w = img_f.shape[:2]
        
        # Get mode parameters
        params = self._get_mode_params(mode)
        
        # Resize signature
        resized_sig = self._resize_signature(h, w)
        
        # Initial detection
        detection_before = None
        if verify and self.extractor is not None:
            result = self.extractor.detect_array(image)
            detection_before = {
                'is_watermarked': result.is_watermarked,
                'confidence': result.confidence,
                'phase_match': result.phase_match
            }
        
        # Apply removal pipeline
        current = img_f.copy()
        method_parts = []
        
        # Step 1: JPEG compression (if first)
        if params.get('jpeg_first', False):
            current = self._jpeg_compress(current, quality=params['jpeg_quality'])
            method_parts.append(f"JPEG_Q{params['jpeg_quality']}")
        
        # Step 2: Signature subtraction
        if params['subtract_scale'] > 0:
            current = current - resized_sig * params['subtract_scale']
            current = np.clip(current, 0, 255)
            method_parts.append(f"Sub_{params['subtract_scale']}x")
        
        # Step 3: JPEG compression (if after subtraction)
        if params.get('jpeg_after', False):
            current = self._jpeg_compress(current, quality=params['jpeg_quality'])
            method_parts.append(f"JPEG_Q{params['jpeg_quality']}")
        
        # Step 4: Additional JPEG passes
        for _ in range(params.get('extra_jpeg_passes', 0)):
            q = params.get('extra_jpeg_quality', 60)
            current = self._jpeg_compress(current, quality=q)
            method_parts.append(f"JPEG_Q{q}")
        
        # Final cleanup
        cleaned = np.clip(current, 0, 255).astype(np.uint8)
        
        # Quality metrics
        psnr = self.compute_psnr(image, cleaned)
        ssim = self.compute_ssim(image, cleaned)
        
        # Final detection
        detection_after = None
        if verify and self.extractor is not None:
            result = self.extractor.detect_array(cleaned)
            detection_after = {
                'is_watermarked': result.is_watermarked,
                'confidence': result.confidence,
                'phase_match': result.phase_match
            }
        
        # Determine success
        success = psnr > 28
        if detection_before and detection_after:
            phase_drop = detection_before['phase_match'] - detection_after['phase_match']
            success = success and (phase_drop > 0.05 or not detection_after['is_watermarked'])
        
        method = ' + '.join(method_parts)
        
        return RemovalResult(
            success=success,
            cleaned_image=cleaned,
            psnr=psnr,
            ssim=ssim,
            detection_before=detection_before,
            detection_after=detection_after,
            method=method,
            details={'mode': mode, 'params': params}
        )
    
    def _remove_combined_worst(
        self,
        image: np.ndarray,
        verify: bool = True,
        strength: str = 'aggressive'
    ) -> RemovalResult:
        """
        Combined worst-case removal using bypass_v2 pipeline.
        
        This is the v2 approach that stacks transforms from multiple
        categories (spatial, quality, noise, color, overlay) to exploit
        SynthID's weakness against combined transforms.
        """
        from synthid_bypass import SynthIDBypass
        
        bypass = SynthIDBypass(extractor=self.extractor)
        result = bypass.bypass_v2(image, strength=strength, verify=verify)
        
        return RemovalResult(
            success=result.success,
            cleaned_image=result.cleaned_image,
            psnr=result.psnr,
            ssim=result.ssim,
            detection_before=result.detection_before,
            detection_after=result.detection_after,
            method=f'combined_worst_{strength}',
            details={
                'mode': 'combined_worst',
                'strength': strength,
                'stages': result.stages_applied,
                'v2_details': result.details
            }
        )
    
    def _get_mode_params(self, mode: str) -> Dict:
        """Get parameters for each removal mode."""
        if mode == 'light':
            return {
                'subtract_scale': 0.5,
                'jpeg_first': False,
                'jpeg_after': True,
                'jpeg_quality': 65,
                'extra_jpeg_passes': 0,
            }
        elif mode == 'aggressive':
            return {
                'subtract_scale': 2.0,
                'jpeg_first': True,
                'jpeg_after': True,
                'jpeg_quality': 50,
                'extra_jpeg_passes': 0,
            }
        elif mode == 'maximum':
            return {
                'subtract_scale': 5.0,
                'jpeg_first': True,
                'jpeg_after': True,
                'jpeg_quality': 50,
                'extra_jpeg_passes': 1,
                'extra_jpeg_quality': 55,
            }
        else:  # balanced (default)
            return {
                'subtract_scale': 1.0,
                'jpeg_first': True,
                'jpeg_after': False,
                'jpeg_quality': 50,
                'extra_jpeg_passes': 0,
            }
    
    def remove_file(
        self,
        input_path: str,
        output_path: str,
        mode: str = 'balanced',
        verify: bool = True,
        strength: str = 'aggressive'
    ) -> RemovalResult:
        """Remove watermark from image file and save result."""
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.remove(img_rgb, mode=mode, verify=verify, strength=strength)
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(result.cleaned_image, cv2.COLOR_RGB2BGR))
        
        return result
    
    def batch_remove(
        self,
        input_dir: str,
        output_dir: str,
        mode: str = 'balanced',
        verify: bool = True,
        limit: int = None,
        strength: str = 'aggressive'
    ):
        """Remove watermark from all images in a directory."""
        import glob
        
        os.makedirs(output_dir, exist_ok=True)
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(input_dir, ext)))
        files = sorted(files)
        
        if limit:
            files = files[:limit]
        
        print(f"Processing {len(files)} images in {mode} mode")
        if mode == 'combined_worst':
            print(f"Strength: {strength}")
        print("=" * 70)
        
        results = []
        for i, f in enumerate(files):
            basename = os.path.basename(f)
            output_path = os.path.join(output_dir, basename)
            
            try:
                result = self.remove_file(f, output_path, mode=mode, verify=verify, strength=strength)
                results.append(result)
                
                if verify and result.detection_before and result.detection_after:
                    before = result.detection_before['phase_match']
                    after = result.detection_after['phase_match']
                    drop = (before - after) / before * 100
                    det_before = '✓' if result.detection_before['is_watermarked'] else '✗'
                    det_after = '✓' if result.detection_after['is_watermarked'] else '✗'
                    print(f"  [{i+1}/{len(files)}] {basename:20s} | {det_before}→{det_after} | "
                          f"phase: {before:.3f}→{after:.3f} ({drop:+5.1f}%) | PSNR: {result.psnr:.1f}dB")
                else:
                    print(f"  [{i+1}/{len(files)}] {basename:20s} | PSNR: {result.psnr:.1f}dB")
            except Exception as e:
                print(f"  [{i+1}/{len(files)}] {basename:20s} | ERROR: {e}")
        
        # Summary
        if results and verify:
            drops = []
            successes = 0
            for r in results:
                if r.detection_before and r.detection_after:
                    before = r.detection_before['phase_match']
                    after = r.detection_after['phase_match']
                    drops.append((before - after) / before * 100)
                    if not r.detection_after['is_watermarked']:
                        successes += 1
            
            print("=" * 70)
            print(f"Results: {len(results)} images processed")
            if drops:
                print(f"  Average phase drop: {np.mean(drops):.1f}%")
                print(f"  Best phase drop: {max(drops):.1f}%")
                print(f"  Undetected: {successes}/{len(results)}")
            print(f"  Average PSNR: {np.mean([r.psnr for r in results]):.1f}dB")
        
        return results


# ================================================================
# CLI INTERFACE
# ================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SynthID Watermark Remover (Signature-Based)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Remove watermark from a single image
  python watermark_remover.py remove input.png output.png --signature artifacts/signature/

  # Batch remove from directory
  python watermark_remover.py batch /path/to/images/ /path/to/output/ --signature artifacts/signature/

  # Extract signature from pure images
  python watermark_remover.py extract --black assets/black/gemini/ --white assets/white/gemini/ -o artifacts/signature/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove watermark from image')
    remove_parser.add_argument('input', help='Input image path')
    remove_parser.add_argument('output', help='Output image path')
    remove_parser.add_argument('--signature', '-s', default='artifacts/signature/',
                               help='Path to signature directory')
    remove_parser.add_argument('--mode', '-m', default='balanced',
                               choices=['light', 'balanced', 'aggressive', 'maximum', 'combined_worst'],
                               help='Removal mode')
    remove_parser.add_argument('--strength', default='aggressive',
                               choices=['moderate', 'aggressive', 'maximum'],
                               help='Strength for combined_worst mode')
    remove_parser.add_argument('--codebook', '-c', default=None,
                               help='Codebook path for verification')
    remove_parser.add_argument('--no-verify', action='store_true',
                               help='Skip verification')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch remove watermarks')
    batch_parser.add_argument('input_dir', help='Input directory')
    batch_parser.add_argument('output_dir', help='Output directory')
    batch_parser.add_argument('--signature', '-s', default='artifacts/signature/')
    batch_parser.add_argument('--mode', '-m', default='balanced',
                              choices=['light', 'balanced', 'aggressive', 'maximum', 'combined_worst'])
    batch_parser.add_argument('--strength', default='aggressive',
                               choices=['moderate', 'aggressive', 'maximum'])
    batch_parser.add_argument('--codebook', '-c', default=None)
    batch_parser.add_argument('--no-verify', action='store_true')
    batch_parser.add_argument('--limit', '-n', type=int, default=None)
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract signature from pure images')
    extract_parser.add_argument('--black', help='Directory of pure black Gemini images')
    extract_parser.add_argument('--white', help='Directory of pure white Gemini images')
    extract_parser.add_argument('-o', '--output', default='artifacts/signature/',
                               help='Output directory for signature')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'extract':
        remover = WatermarkRemover()
        remover.extract_signature_from_images(
            black_dir=args.black,
            white_dir=args.white,
            output_dir=args.output
        )
    else:
        # Load extractor for verification
        extractor = None
        codebook = getattr(args, 'codebook', None)
        no_verify = getattr(args, 'no_verify', False)
        
        if codebook and not no_verify:
            try:
                from robust_extractor import RobustSynthIDExtractor
                extractor = RobustSynthIDExtractor()
                extractor.load_codebook(codebook)
            except Exception as e:
                print(f"Warning: Could not load extractor: {e}")
        
        sig_dir = args.signature
        remover = WatermarkRemover(signature_dir=sig_dir, extractor=extractor)
        strength = getattr(args, 'strength', 'aggressive')
        
        if args.command == 'remove':
            result = remover.remove_file(
                args.input, args.output,
                mode=args.mode, verify=not no_verify,
                strength=strength
            )
            
            print("\n" + "=" * 60)
            print("WATERMARK REMOVAL RESULTS")
            print("=" * 60)
            print(f"  Mode: {args.mode}")
            if args.mode == 'combined_worst':
                print(f"  Strength: {strength}")
            print(f"  Method: {result.method}")
            print(f"  Success: {result.success}")
            print(f"  PSNR: {result.psnr:.2f} dB")
            print(f"  SSIM: {result.ssim:.4f}")
            
            if result.detection_before:
                print(f"\n  Before:")
                print(f"    Watermarked: {result.detection_before['is_watermarked']}")
                print(f"    Phase Match: {result.detection_before['phase_match']:.4f}")
            
            if result.detection_after:
                print(f"\n  After:")
                print(f"    Watermarked: {result.detection_after['is_watermarked']}")
                print(f"    Phase Match: {result.detection_after['phase_match']:.4f}")
                
                if result.detection_before:
                    drop = result.detection_before['phase_match'] - result.detection_after['phase_match']
                    pct = 100 * drop / result.detection_before['phase_match']
                    print(f"\n  Phase Drop: {drop:.4f} ({pct:.1f}%)")
            
            print("=" * 60)
            print(f"Saved to: {args.output}")
        
        elif args.command == 'batch':
            remover.batch_remove(
                args.input_dir, args.output_dir,
                mode=args.mode, verify=not no_verify,
                limit=args.limit,
                strength=strength
            )
