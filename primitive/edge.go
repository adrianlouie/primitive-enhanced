package primitive

import (
	"image"
	"math"
	"math/rand"
)

// ComputeEdgeMap returns a Sobel edge magnitude map normalized to [0, 1].
func ComputeEdgeMap(im *image.RGBA) [][]float64 {
	size := im.Bounds().Size()
	w, h := size.X, size.Y

	gray := make([][]float64, h)
	for y := 0; y < h; y++ {
		gray[y] = make([]float64, w)
		for x := 0; x < w; x++ {
			i := im.PixOffset(x, y)
			gray[y][x] = (float64(im.Pix[i]) + float64(im.Pix[i+1]) + float64(im.Pix[i+2])) / 3.0
		}
	}

	edge := make([][]float64, h)
	maxMag := 0.0
	for y := 0; y < h; y++ {
		edge[y] = make([]float64, w)
		if y == 0 || y == h-1 {
			continue
		}
		for x := 1; x < w-1; x++ {
			gx := -gray[y-1][x-1] + gray[y-1][x+1] -
				2*gray[y][x-1] + 2*gray[y][x+1] -
				gray[y+1][x-1] + gray[y+1][x+1]
			gy := -gray[y-1][x-1] - 2*gray[y-1][x] - gray[y-1][x+1] +
				gray[y+1][x-1] + 2*gray[y+1][x] + gray[y+1][x+1]
			mag := math.Sqrt(gx*gx + gy*gy)
			edge[y][x] = mag
			if mag > maxMag {
				maxMag = mag
			}
		}
	}

	if maxMag > 0 {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				edge[y][x] /= maxMag
			}
		}
	}
	return edge
}

// ComputeErrorMap returns per-pixel squared RGB error.
func ComputeErrorMap(target, current *image.RGBA) [][]float64 {
	size := target.Bounds().Size()
	w, h := size.X, size.Y
	errMap := make([][]float64, h)
	for y := 0; y < h; y++ {
		errMap[y] = make([]float64, w)
		for x := 0; x < w; x++ {
			i := target.PixOffset(x, y)
			dr := float64(target.Pix[i]) - float64(current.Pix[i])
			dg := float64(target.Pix[i+1]) - float64(current.Pix[i+1])
			db := float64(target.Pix[i+2]) - float64(current.Pix[i+2])
			errMap[y][x] = dr*dr + dg*dg + db*db
		}
	}
	return errMap
}

// UpdateErrorMapRegion updates the error map for a rectangular region.
func UpdateErrorMapRegion(errMap [][]float64, target, current *image.RGBA, x1, y1, x2, y2 int) {
	for y := y1; y < y2; y++ {
		for x := x1; x < x2; x++ {
			i := target.PixOffset(x, y)
			dr := float64(target.Pix[i]) - float64(current.Pix[i])
			dg := float64(target.Pix[i+1]) - float64(current.Pix[i+1])
			db := float64(target.Pix[i+2]) - float64(current.Pix[i+2])
			errMap[y][x] = dr*dr + dg*dg + db*db
		}
	}
}

// WeightedRandomPoint samples (x, y) proportional to map values.
func WeightedRandomPoint(m [][]float64, rnd *rand.Rand) (int, int) {
	h := len(m)
	if h == 0 {
		return 0, 0
	}
	w := len(m[0])

	var total float64
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			total += m[y][x]
		}
	}
	if total < 1e-9 {
		return rnd.Intn(w), rnd.Intn(h)
	}

	r := rnd.Float64() * total
	cum := 0.0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			cum += m[y][x]
			if cum >= r {
				return x, y
			}
		}
	}
	return w / 2, h / 2
}
