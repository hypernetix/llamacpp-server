package main

import (
	"fmt"
	"os"

	llamacppbindings "github.com/hypernetix/llamacpp_server/internal/bindings"
	"github.com/hypernetix/llamacpp_server/internal/logging"
)

func main() {
	args := os.Args[1:]
	if len(args) != 1 {
		fmt.Println("Usage: inferencetest1 <model_path>")
		return
	}

	modelPath := args[0]

	arch, err := llamacppbindings.GetModelArch(modelPath)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("Model architecture: %s\n", arch)

	prompt := "What is a capital of USA?"
	nPredict := 32 // generate 32 tokens

	logger := logging.NewSprintfLogger()

	llamacppbindings.Initialize(logger)

	modelParams := llamacppbindings.NewModelDefaultParams()
	modelParams.SetNGpuLayers(99)
	modelParams.SetUseMmap(false)
	modelParams.SetProgressCallback(func(progress float32) {
		fmt.Printf("progress: %f\n", progress)
	})

	defer modelParams.Free()

	model, err := llamacppbindings.LoadModelFromFile(modelPath, modelParams)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer model.Free()

	fmt.Printf("Model info: %+v\n", model.Info())

	vocab := model.Vocab()

	fmt.Printf("Vocab info: %+v\n", vocab.Info())

	tokens, err := vocab.Tokenize(prompt, true, true)
	if err != nil {
		fmt.Println(err)
		return
	}

	contextParams := llamacppbindings.NewContextDefaultParams()
	contextParams.SetNCtx(len(tokens) + nPredict)
	contextParams.SetNBatch(len(tokens))

	context, err := llamacppbindings.NewContext(model, contextParams)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer context.Free()

	samplerChainParams := llamacppbindings.NewSamplerChainDefaultParams()
	samplerChain, err := llamacppbindings.NewSamplerChain(samplerChainParams)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer samplerChain.Free()

	samplerGreedy, err := llamacppbindings.NewGreedySampler()
	if err != nil {
		fmt.Println(err)
		return
	}
	samplerChain.AddSampler(samplerGreedy)

	sampler := samplerChain.Sampler()

	fmt.Printf("Prompt: %s\n", prompt)
	fmt.Printf("Tokens: %+v\n", tokens)

	fmt.Printf("Token pieces:\n")

	for _, token := range tokens {
		piece, err := vocab.TokenToPiece(token)
		if err != nil {
			fmt.Println(err)
			return
		}
		fmt.Printf("%s\n", piece)
	}

	batch := llamacppbindings.NotOwnedOneItemBatch(tokens) // must not be freed
	var nextToken int

	pos := 0
	decoded := 0

	//fmt.Printf("pos: %+v, batch.NTokens(): %+v, len(tokens): %+v, nPredict: %+v\n", pos, batch.NTokens(), len(tokens), nPredict)

	for pos+batch.NTokens() < len(tokens)+nPredict {
		if err := context.Decode(batch); err != nil {
			fmt.Println(err)
			return
		}

		pos += batch.NTokens()

		nextToken = sampler.Sample(context, -1)

		// is it an end of generation?
		if vocab.IsEog(nextToken) {
			break
		}

		piece, err := vocab.TokenToPiece(nextToken)
		if err != nil {
			fmt.Println(err)
			return
		}
		fmt.Printf("%s\n", piece)

		batch = llamacppbindings.NotOwnedOneItemBatch([]int{nextToken})

		decoded += 1

		//fmt.Printf("pos: %+v, batch.NTokens(): %+v, len(tokens): %+v, nPredict: %+v\n", pos, batch.NTokens(), len(tokens), nPredict)
	}

	fmt.Println()
	fmt.Printf("Decoded: %d\n", decoded)
}
