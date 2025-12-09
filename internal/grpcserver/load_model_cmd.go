package llamacppgrpcserver

import (
	llamacppbindings "github.com/hypernetix/llamacpp_server/internal/bindings"
	"github.com/hypernetix/llamacpp_server/internal/logging"
	"github.com/hypernetix/llamacpp_server/internal/modelmanagement"
)

type LoadModelOptions struct {
	NGpuLayers int
	UseMmap    bool
}

type ModelData struct {
	ModelParams *llamacppbindings.ModelParams
	Model       *llamacppbindings.Model
}

func (md *ModelData) Destroy() error {
	if md.Model != nil {
		md.Model.Free()
	}
	return nil
}

func NewLoadModelFunc(options LoadModelOptions, logger logging.SprintfLogger) modelmanagement.LoadModelFunc {
	cmd := &loadModelCmd{
		options: options,
		logger:  logger.With("module", "loadModelCmd"),
	}
	return cmd.Do
}

type loadModelCmd struct {
	options LoadModelOptions
	logger  logging.SprintfLogger
}

func (cmd *loadModelCmd) Do(path string, progress modelmanagement.LoadModelProgressFunc) (interface{}, error) {
	cmd.logger.Debugf("Do: %s", path)

	modelParams := llamacppbindings.NewModelDefaultParams()
	modelParams.SetNGpuLayers(cmd.options.NGpuLayers)
	modelParams.SetUseMmap(cmd.options.UseMmap)
	modelParams.SetProgressCallback(progress)

	cmd.logger.Debugf("Do: modelParams: %+v", modelParams)

	model, err := llamacppbindings.LoadModelFromFile(path, modelParams)
	if err != nil {
		return nil, err
	}

	modelData := &ModelData{
		ModelParams: modelParams,
		Model:       model,
	}

	cmd.logger.Debugf("Do: model loaded, info: %+v", model.Info())
	return modelData, nil
}
