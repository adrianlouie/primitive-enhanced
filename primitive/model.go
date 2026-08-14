package primitive

import (
	"fmt"
	"image"
	"image/draw"
	"strings"

	"github.com/fogleman/gg"
)

type Model struct {
	Sw, Sh     int
	Scale      float64
	Background Color
	Target     *image.RGBA
	Current    *image.RGBA
	Context    *gg.Context
	Score      float64
	Shapes     []Shape
	Colors     []Color
	Scores     []float64
	Workers    []*Worker

	// Enhanced features
	UseLab      bool
	TargetLab   *LabBuffer
	EdgeMap     [][]float64
	ErrorMap    [][]float64
	TotalShapes int
	StepCount   int
}

func NewModel(target image.Image, background Color, size, numWorkers int) *Model {
	w := target.Bounds().Size().X
	h := target.Bounds().Size().Y
	aspect := float64(w) / float64(h)
	var sw, sh int
	var scale float64
	if aspect >= 1 {
		sw = size
		sh = int(float64(size) / aspect)
		scale = float64(size) / float64(w)
	} else {
		sw = int(float64(size) * aspect)
		sh = size
		scale = float64(size) / float64(h)
	}

	model := &Model{}
	model.Sw = sw
	model.Sh = sh
	model.Scale = scale
	model.Background = background
	model.Target = imageToRGBA(target)
	model.Current = uniformRGBA(target.Bounds(), background.NRGBA())
	model.Context = model.newContext()

	// Default to Lab scoring
	model.UseLab = true
	model.TargetLab = NewLabBuffer(model.Target)

	// Compute initial score
	if model.UseLab {
		model.Score = differenceFullLab(model.TargetLab, model.Current)
	} else {
		model.Score = differenceFull(model.Target, model.Current)
	}

	// Precompute edge map and error map
	model.EdgeMap = ComputeEdgeMap(model.Target)
	model.ErrorMap = ComputeErrorMap(model.Target, model.Current)

	for i := 0; i < numWorkers; i++ {
		worker := NewWorker(model.Target)
		worker.TargetLab = model.TargetLab
		worker.UseLab = model.UseLab
		model.Workers = append(model.Workers, worker)
	}
	return model
}

// SetUseLab toggles Lab vs RGB scoring for the model and all its workers,
// recomputing Score from scratch so it stays on the correct scale — Lab
// scores (differenceFullLab) and RGB scores (differenceFull) are not
// interchangeable, since only the RGB formula is normalized by /255.
func (model *Model) SetUseLab(useLab bool) {
	model.UseLab = useLab
	for _, w := range model.Workers {
		w.UseLab = useLab
	}
	if useLab {
		model.Score = differenceFullLab(model.TargetLab, model.Current)
	} else {
		model.Score = differenceFull(model.Target, model.Current)
	}
}

func (model *Model) newContext() *gg.Context {
	dc := gg.NewContext(model.Sw, model.Sh)
	dc.Scale(model.Scale, model.Scale)
	dc.Translate(0.5, 0.5)
	dc.SetColor(model.Background.NRGBA())
	dc.Clear()
	return dc
}

func (model *Model) Frames(scoreDelta float64) []image.Image {
	var result []image.Image
	dc := model.newContext()
	result = append(result, imageToRGBA(dc.Image()))
	previous := 10.0
	for i, shape := range model.Shapes {
		c := model.Colors[i]
		dc.SetRGBA255(c.R, c.G, c.B, c.A)
		shape.Draw(dc, model.Scale)
		dc.Fill()
		score := model.Scores[i]
		delta := previous - score
		if delta >= scoreDelta {
			previous = score
			result = append(result, imageToRGBA(dc.Image()))
		}
	}
	return result
}

func (model *Model) SVG() string {
	bg := model.Background
	var lines []string
	lines = append(lines, fmt.Sprintf("<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"%d\" height=\"%d\">", model.Sw, model.Sh))
	lines = append(lines, fmt.Sprintf("<rect x=\"0\" y=\"0\" width=\"%d\" height=\"%d\" fill=\"#%02x%02x%02x\" />", model.Sw, model.Sh, bg.R, bg.G, bg.B))
	lines = append(lines, fmt.Sprintf("<g transform=\"scale(%f) translate(0.5 0.5)\">", model.Scale))
	for i, shape := range model.Shapes {
		c := model.Colors[i]
		attrs := "fill=\"#%02x%02x%02x\" fill-opacity=\"%f\""
		attrs = fmt.Sprintf(attrs, c.R, c.G, c.B, float64(c.A)/255)
		lines = append(lines, shape.SVG(attrs))
	}
	lines = append(lines, "</g>")
	lines = append(lines, "</svg>")
	return strings.Join(lines, "\n")
}

func (model *Model) Add(shape Shape, alpha int) {
	before := copyRGBA(model.Current)
	lines := shape.Rasterize()

	var color Color
	if model.UseLab {
		color = solveColorAlpha(model.Target, model.Current, lines, alpha)
	} else {
		color = computeColor(model.Target, model.Current, lines, alpha)
	}
	drawLines(model.Current, color, lines)

	var score float64
	if model.UseLab {
		score = differencePartialLab(model.TargetLab, before, model.Current, model.Score, lines)
	} else {
		score = differencePartial(model.Target, before, model.Current, model.Score, lines)
	}

	model.Score = score
	model.Shapes = append(model.Shapes, shape)
	model.Colors = append(model.Colors, color)
	model.Scores = append(model.Scores, score)

	model.Context.SetRGBA255(color.R, color.G, color.B, color.A)
	shape.Draw(model.Context, model.Scale)

	// Incrementally update error map for the bounding box of changed scanlines
	if model.ErrorMap != nil && len(lines) > 0 {
		w := model.Target.Bounds().Size().X
		h := model.Target.Bounds().Size().Y
		minX, minY := w, h
		maxX, maxY := 0, 0
		for _, line := range lines {
			if line.Y < minY {
				minY = line.Y
			}
			if line.Y > maxY {
				maxY = line.Y
			}
			if line.X1 < minX {
				minX = line.X1
			}
			if line.X2 > maxX {
				maxX = line.X2
			}
		}
		minX = maxInt(0, minX)
		minY = maxInt(0, minY)
		maxX = minInt(w, maxX+1)
		maxY = minInt(h, maxY+1)
		UpdateErrorMapRegion(model.ErrorMap, model.Target, model.Current, minX, minY, maxX, maxY)
	}
}

func (model *Model) Step(shapeType ShapeType, alpha, repeat int) int {
	model.StepCount++
	progress := float64(model.StepCount) / float64(model.TotalShapes)

	// 3-phase annealing for shape size
	tw := model.Target.Bounds().Size().X
	th := model.Target.Bounds().Size().Y
	maxDim := maxInt(tw, th)
	var maxSize int
	if progress < 0.40 {
		t := progress / 0.40
		maxSize = int(float64(maxDim) * (0.60 - 0.35*t))
	} else if progress < 0.75 {
		t := (progress - 0.40) / 0.35
		maxSize = int(float64(maxDim) * (0.25 - 0.15*t))
	} else {
		t := (progress - 0.75) / 0.25
		maxSize = int(float64(maxDim) * (0.10 - 0.07*t))
	}
	maxSize = maxInt(4, maxSize)

	// Sample focus points from error map and edge map
	rnd := model.Workers[0].Rnd
	fx, fy := WeightedRandomPoint(model.ErrorMap, rnd)
	ex, ey := WeightedRandomPoint(model.EdgeMap, rnd)

	// Configure all workers
	for _, worker := range model.Workers {
		worker.MaxSize = maxSize
		worker.FocusX = fx
		worker.FocusY = fy
		worker.HasFocus = true
		worker.EdgeFocusX = ex
		worker.EdgeFocusY = ey
		worker.HasEdgeFocus = true
	}

	state := model.runWorkers(shapeType, alpha, 1000, 100, 16)

	// Winner refinement: 30 extra hill climb mutations on the best
	state = HillClimb(state, 30).(*State)

	model.Add(state.Shape, state.Alpha)

	for i := 0; i < repeat; i++ {
		state.Worker.Init(model.Current, model.Score)
		a := state.Energy()
		state = HillClimb(state, 100).(*State)
		b := state.Energy()
		if a == b {
			break
		}
		model.Add(state.Shape, state.Alpha)
	}

	counter := 0
	for _, worker := range model.Workers {
		counter += worker.Counter
	}
	return counter
}

func (model *Model) runWorkers(t ShapeType, a, n, age, m int) *State {
	wn := len(model.Workers)
	ch := make(chan *State, wn)
	wm := m / wn
	if m%wn != 0 {
		wm++
	}
	for i := 0; i < wn; i++ {
		worker := model.Workers[i]
		worker.Init(model.Current, model.Score)
		go model.runWorker(worker, t, a, n, age, wm, ch)
	}
	var bestEnergy float64
	var bestState *State
	for i := 0; i < wn; i++ {
		state := <-ch
		energy := state.Energy()
		if i == 0 || energy < bestEnergy {
			bestEnergy = energy
			bestState = state
		}
	}
	return bestState
}

func (model *Model) runWorker(worker *Worker, t ShapeType, a, n, age, m int, ch chan *State) {
	ch <- worker.BestHillClimbState(t, a, n, age, m)
}

// SaveComparison saves a side-by-side comparison (original | reconstruction).
func (model *Model) SaveComparison(path string) error {
	tw := model.Target.Bounds().Size().X
	th := model.Target.Bounds().Size().Y
	gap := 4
	comp := image.NewRGBA(image.Rect(0, 0, tw*2+gap, th))
	draw.Draw(comp, image.Rect(0, 0, tw, th), model.Target, image.Point{}, draw.Src)
	draw.Draw(comp, image.Rect(tw+gap, 0, tw*2+gap, th), model.Current, image.Point{}, draw.Src)
	return SavePNG(path, comp)
}
