package llmservice

import (
	"fmt"

	"github.com/hypernetix/llamacpp_server/internal/inferenceengine"
	"github.com/hypernetix/llamacpp_server/internal/logging"
	"github.com/hypernetix/llamacpp_server/internal/modelmanagement"
)

type PredictOptions struct {
	FlashAttn     bool
	NParallel     int
	NThreads      int
	NThreadsBatch int
	CtxSize       int
	BatchSize     int
}

type Options struct {
	Model   LoadModelOptions
	Predict PredictOptions
}

type Service struct {
	modelManager       modelmanagement.ModelManager
	predictionsManager inferenceengine.PredictionsManager
	logger             logging.SprintfLogger
}

func NewService(opts Options, logger logging.SprintfLogger) *Service {
	loadModelFunc := newLoadModelFunc(opts.Model, logger)
	modelMgr := modelmanagement.NewModelManager(loadModelFunc, logger)

	nParallel := opts.Predict.NParallel
	if nParallel <= 0 {
		nParallel = 1
	}

	predictionsMgr := inferenceengine.New(inferenceengine.Options{
		NParallel:     nParallel,
		CtxSize:       opts.Predict.CtxSize,
		BatchSize:     opts.Predict.BatchSize,
		NThreads:      opts.Predict.NThreads,
		NThreadsBatch: opts.Predict.NThreadsBatch,
		FlashAttn:     opts.Predict.FlashAttn,
	}, logger)
	logger.Infof("continuous batching enabled (slots=%d)", nParallel)

	return &Service{
		modelManager:       modelMgr,
		predictionsManager: predictionsMgr,
		logger:             logger.With("module", "llmservice.Service"),
	}
}

func (s *Service) LoadModel(path string, onProgress func(float32)) error {
	s.logger.Debugf("LoadModel: %s", path)
	model, err := s.modelManager.LoadModel(path, onProgress)
	if err != nil {
		return err
	}
	md, ok := model.(*ModelData)
	if !ok {
		return fmt.Errorf("invalid model type")
	}
	s.logger.Debugf("LoadModel: loaded, params: %+v, info: %+v",
		md.ModelParams, md.Model.Info())
	return nil
}

func (s *Service) Predict(modelPath string, prompt string, args inferenceengine.PredictArgs, stream inferenceengine.StreamFunc) (string, error) {
	model, err := s.modelManager.GetModel(modelPath)
	if err != nil {
		return "", err
	}
	md, ok := model.(*ModelData)
	if !ok {
		return "", fmt.Errorf("invalid model type")
	}
	return s.predictionsManager.Predict(md.Model, prompt, args, stream)
}

func (s *Service) ListModels() []string {
	return s.modelManager.ListModels()
}

func (s *Service) Stop() {
	s.predictionsManager.Stop()
	s.modelManager.Stop()
}
