package llmservice

import (
	"fmt"

	"github.com/hypernetix/llamacpp_server/internal/inference"
	"github.com/hypernetix/llamacpp_server/internal/logging"
	"github.com/hypernetix/llamacpp_server/internal/modelmanagement"
)

type Options struct {
	Model   LoadModelOptions
	Predict PredictOptions
}

type Service struct {
	modelManager       modelmanagement.ModelManager
	predictionsManager inference.PredictionsManager
	logger             logging.SprintfLogger
}

func NewService(opts Options, logger logging.SprintfLogger) *Service {
	loadModelFunc := newLoadModelFunc(opts.Model, logger)
	modelMgr := modelmanagement.NewModelManager(loadModelFunc, logger)
	predictFunc := newPredictFunc(opts.Predict, logger)
	predictionsMgr := inference.NewPredictionsManager(predictFunc, opts.Predict.NParallel)
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

func (s *Service) Predict(modelPath string, prompt string, args inference.PredictArgs, stream inference.StreamFunc) (string, error) {
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

func (s *Service) Stop() {
	s.predictionsManager.Stop()
	s.modelManager.Stop()
}
