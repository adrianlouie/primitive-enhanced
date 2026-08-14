package primitive

import (
	"image"
	"math"
)

// Precomputed sRGB linearization lookup table — eliminates math.Pow per pixel.
var linearLUT [256]float64

func init() {
	for i := 0; i < 256; i++ {
		v := float64(i) / 255.0
		if v > 0.04045 {
			linearLUT[i] = math.Pow((v+0.055)/1.055, 2.4)
		} else {
			linearLUT[i] = v / 12.92
		}
	}
}

// LabBuffer stores precomputed CIE Lab values for an image.
type LabBuffer struct {
	Data []float64 // L, a, b interleaved (3 values per pixel)
	W, H int
}

func labF(t float64) float64 {
	if t > 0.008856 {
		return math.Cbrt(t)
	}
	return (903.3*t + 16.0) / 116.0
}

// RGBToLab converts sRGB [0-255] to CIE Lab using precomputed LUT.
func RGBToLab(r, g, b uint8) (L, a, bv float64) {
	rf := linearLUT[r]
	gf := linearLUT[g]
	bf := linearLUT[b]

	x := (0.4124564*rf + 0.3575761*gf + 0.1804375*bf) / 0.95047
	y := 0.2126729*rf + 0.7151522*gf + 0.0721750*bf
	z := (0.0193339*rf + 0.1191920*gf + 0.9503041*bf) / 1.08883

	fx := labF(x)
	fy := labF(y)
	fz := labF(z)

	L = 116.0*fy - 16.0
	a = 500.0 * (fx - fy)
	bv = 200.0 * (fy - fz)
	return
}

// NewLabBuffer precomputes Lab values for every pixel of an RGBA image.
func NewLabBuffer(im *image.RGBA) *LabBuffer {
	size := im.Bounds().Size()
	w, h := size.X, size.Y
	data := make([]float64, w*h*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := im.PixOffset(x, y)
			L, a, b := RGBToLab(im.Pix[i], im.Pix[i+1], im.Pix[i+2])
			j := (y*w + x) * 3
			data[j] = L
			data[j+1] = a
			data[j+2] = b
		}
	}
	return &LabBuffer{data, w, h}
}

// At returns the Lab values for pixel (x, y).
func (lb *LabBuffer) At(x, y int) (L, a, b float64) {
	j := (y*lb.W + x) * 3
	return lb.Data[j], lb.Data[j+1], lb.Data[j+2]
}

// differenceFullLab computes RMS Lab error across the entire image.
func differenceFullLab(targetLab *LabBuffer, current *image.RGBA) float64 {
	w, h := targetLab.W, targetLab.H
	var total float64
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := current.PixOffset(x, y)
			tL, ta, tb := targetLab.At(x, y)
			cL, ca, cb := RGBToLab(current.Pix[i], current.Pix[i+1], current.Pix[i+2])
			dL := tL - cL
			da := ta - ca
			db := tb - cb
			total += dL*dL + da*da + db*db
		}
	}
	return math.Sqrt(total / float64(w*h*3))
}

// differencePartialLab incrementally updates Lab-based score.
// Uses precomputed beforeLab to avoid redundant conversions for the "before" image.
func differencePartialLab(targetLab *LabBuffer, before, after *image.RGBA, score float64, lines []Scanline) float64 {
	w, h := targetLab.W, targetLab.H
	total := score * score * float64(w*h*3)
	for _, line := range lines {
		for x := line.X1; x <= line.X2; x++ {
			tL, ta, tb := targetLab.At(x, line.Y)

			i := before.PixOffset(x, line.Y)
			bL, ba, bb := RGBToLab(before.Pix[i], before.Pix[i+1], before.Pix[i+2])

			aL, aa, ab := RGBToLab(after.Pix[i], after.Pix[i+1], after.Pix[i+2])

			total -= (tL-bL)*(tL-bL) + (ta-ba)*(ta-ba) + (tb-bb)*(tb-bb)
			total += (tL-aL)*(tL-aL) + (ta-aa)*(ta-aa) + (tb-ab)*(tb-ab)
		}
	}
	if total < 0 {
		total = 0
	}
	return math.Sqrt(total / float64(w*h*3))
}

// differencePartialLabCached uses a precomputed beforeLab buffer to skip
// half the Lab conversions — only "after" pixels need conversion.
func differencePartialLabCached(targetLab, beforeLab *LabBuffer, after *image.RGBA, score float64, lines []Scanline) float64 {
	w, h := targetLab.W, targetLab.H
	total := score * score * float64(w*h*3)
	for _, line := range lines {
		for x := line.X1; x <= line.X2; x++ {
			tL, ta, tb := targetLab.At(x, line.Y)
			bL, ba, bb := beforeLab.At(x, line.Y)

			i := after.PixOffset(x, line.Y)
			aL, aa, ab := RGBToLab(after.Pix[i], after.Pix[i+1], after.Pix[i+2])

			total -= (tL-bL)*(tL-bL) + (ta-ba)*(ta-ba) + (tb-bb)*(tb-bb)
			total += (tL-aL)*(tL-aL) + (ta-aa)*(ta-aa) + (tb-ab)*(tb-ab)
		}
	}
	if total < 0 {
		total = 0
	}
	return math.Sqrt(total / float64(w*h*3))
}
