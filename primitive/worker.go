package primitive

import (
	"image"
	"math/rand"
	"time"

	"github.com/golang/freetype/raster"
)

type Worker struct {
	W, H       int
	Target     *image.RGBA
	Current    *image.RGBA
	Buffer     *image.RGBA
	Rasterizer *raster.Rasterizer
	Lines      []Scanline
	Heatmap    *Heatmap
	Rnd        *rand.Rand
	Score      float64
	Counter    int

	// Enhanced features
	TargetLab    *LabBuffer
	CurrentLab   *LabBuffer // Precomputed once per step to avoid repeated Lab conversion
	UseLab       bool
	MaxSize      int // Annealing-controlled max shape size (0 = use default)
	FocusX       int // Error-weighted focus point
	FocusY       int
	HasFocus     bool
	EdgeFocusX   int // Edge-weighted focus point
	EdgeFocusY   int
	HasEdgeFocus bool
}

func NewWorker(target *image.RGBA) *Worker {
	w := target.Bounds().Size().X
	h := target.Bounds().Size().Y
	worker := Worker{}
	worker.W = w
	worker.H = h
	worker.Target = target
	worker.Buffer = image.NewRGBA(target.Bounds())
	worker.Rasterizer = raster.NewRasterizer(w, h)
	worker.Lines = make([]Scanline, 0, 4096)
	worker.Heatmap = NewHeatmap(w, h)
	worker.Rnd = rand.New(rand.NewSource(time.Now().UnixNano()))
	return &worker
}

func (worker *Worker) Init(current *image.RGBA, score float64) {
	worker.Current = current
	worker.Score = score
	worker.Counter = 0
	worker.Heatmap.Clear()
	// Precompute Lab for current canvas once per step
	if worker.UseLab {
		worker.CurrentLab = NewLabBuffer(current)
	}
}

func (worker *Worker) Energy(shape Shape, alpha int) float64 {
	worker.Counter++
	lines := shape.Rasterize()

	var color Color
	if worker.UseLab {
		color = solveColorAlpha(worker.Target, worker.Current, lines, alpha)
	} else {
		color = computeColor(worker.Target, worker.Current, lines, alpha)
	}

	copyLines(worker.Buffer, worker.Current, lines)
	drawLines(worker.Buffer, color, lines)

	if worker.UseLab && worker.TargetLab != nil && worker.CurrentLab != nil {
		return differencePartialLabCached(worker.TargetLab, worker.CurrentLab, worker.Buffer, worker.Score, lines)
	}
	return differencePartial(worker.Target, worker.Current, worker.Buffer, worker.Score, lines)
}

func (worker *Worker) BestHillClimbState(t ShapeType, a, n, age, m int) *State {
	var bestEnergy float64
	var bestState *State
	for i := 0; i < m; i++ {
		state := worker.BestRandomState(t, a, n)
		before := state.Energy()
		state = HillClimb(state, age).(*State)
		energy := state.Energy()
		vv("%dx random: %.6f -> %dx hill climb: %.6f\n", n, before, age, energy)
		if i == 0 || energy < bestEnergy {
			bestEnergy = energy
			bestState = state
		}
	}
	return bestState
}

func (worker *Worker) BestRandomState(t ShapeType, a, n int) *State {
	var bestEnergy float64
	var bestState *State
	for i := 0; i < n; i++ {
		state := worker.RandomState(t, a)
		energy := state.Energy()
		if i == 0 || energy < bestEnergy {
			bestEnergy = energy
			bestState = state
		}
	}
	return bestState
}

func (worker *Worker) RandomState(t ShapeType, a int) *State {
	switch t {
	default:
		return worker.RandomState(ShapeType(worker.Rnd.Intn(8)+1), a)
	case ShapeTypeTriangle:
		return NewState(worker, NewRandomTriangle(worker), a)
	case ShapeTypeRectangle:
		return NewState(worker, NewRandomRectangle(worker), a)
	case ShapeTypeEllipse:
		return NewState(worker, NewRandomEllipse(worker), a)
	case ShapeTypeCircle:
		return NewState(worker, NewRandomCircle(worker), a)
	case ShapeTypeRotatedRectangle:
		return NewState(worker, NewRandomRotatedRectangle(worker), a)
	case ShapeTypeQuadratic:
		return NewState(worker, NewRandomQuadratic(worker), a)
	case ShapeTypeRotatedEllipse:
		return NewState(worker, NewRandomRotatedEllipse(worker), a)
	case ShapeTypePolygon:
		return NewState(worker, NewRandomPolygon(worker, 4, false), a)
	}
}

// RandomPosition returns a biased random position.
// 60% near error focus, 15% near edge focus, 25% fully random.
func (worker *Worker) RandomPosition() (int, int) {
	r := worker.Rnd.Float64()
	if worker.HasFocus && r < 0.60 {
		spread := worker.MaxSize
		if spread < 20 {
			spread = 20
		}
		x := clampInt(worker.FocusX+worker.Rnd.Intn(spread*2+1)-spread, 0, worker.W-1)
		y := clampInt(worker.FocusY+worker.Rnd.Intn(spread*2+1)-spread, 0, worker.H-1)
		return x, y
	}
	if worker.HasEdgeFocus && r < 0.75 {
		spread := worker.MaxSize
		if spread < 20 {
			spread = 20
		}
		x := clampInt(worker.EdgeFocusX+worker.Rnd.Intn(spread*2+1)-spread, 0, worker.W-1)
		y := clampInt(worker.EdgeFocusY+worker.Rnd.Intn(spread*2+1)-spread, 0, worker.H-1)
		return x, y
	}
	return worker.Rnd.Intn(worker.W), worker.Rnd.Intn(worker.H)
}

// ShapeMaxSize returns the annealing-controlled max size, or a default.
func (worker *Worker) ShapeMaxSize(defaultSize int) int {
	if worker.MaxSize > 0 {
		return worker.MaxSize
	}
	return defaultSize
}
