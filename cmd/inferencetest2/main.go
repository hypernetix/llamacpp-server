package main

import (
	"fmt"
	"os"

	llamacppbindings "github.com/hypernetix/llamacpp_server/internal/bindings"
)

func main() {
	args := os.Args[1:]
	if len(args) != 1 {
		fmt.Println("Usage: test2 <model_path>")
		return
	}

	modelPath := args[0]

	arch, err := llamacppbindings.GetModelArch(modelPath)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("Model architecture: %s\n", arch)

	llamacppbindings.Initialize(nil)

	modelParams := llamacppbindings.NewModelDefaultParams()
	modelParams.SetNGpuLayers(99)
	modelParams.SetUseMmap(false)

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

	contextParams := llamacppbindings.NewContextDefaultParams()
	contextParams.SetNCtx(2048)
	contextParams.SetNBatch(2048)

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

	samplerMinP, err := llamacppbindings.NewMinPSampler(0.05, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	samplerChain.AddSampler(samplerMinP)

	samplerTemp, err := llamacppbindings.NewTempSampler(0.8)
	if err != nil {
		fmt.Println(err)
		return
	}
	samplerChain.AddSampler(samplerTemp)

	samplerSeed, err := llamacppbindings.NewSeedSampler(0xFFFFFFFF)
	if err != nil {
		fmt.Println(err)
		return
	}
	samplerChain.AddSampler(samplerSeed)

	sampler := samplerChain.Sampler()

	generate := func(prompt string) (string, error) {
		isFirst := context.NCellsUsed() == 0

		tokens, err := vocab.Tokenize(prompt, isFirst, true)
		if err != nil {
			return "", err
		}

		fmt.Printf("tokens: %+v\n", tokens)

		generatedText := ""

		batch := llamacppbindings.NotOwnedOneItemBatch(tokens) // must not be freed
		var nextToken int
		for {
			n := context.NCells()
			nUsed := context.NCellsUsed()
			if nUsed+batch.NTokens() > n {
				return "", fmt.Errorf("context size exceeded")
			}

			err := context.Decode(batch)
			if err != nil {
				return "", err
			}

			nextToken = sampler.Sample(context, -1)
			if vocab.IsEog(nextToken) {
				break
			}

			fmt.Printf("nextToken: %+v\n", nextToken)

			piece, err := vocab.TokenToPiece(nextToken)
			if err != nil {
				return "", err
			}

			fmt.Printf("piece: %+v\n", piece)

			generatedText += piece

			batch = llamacppbindings.NotOwnedOneItemBatch([]int{nextToken})
		}

		return generatedText, nil
	}

	prompt := "What is the capital of USA?"
	fmt.Println("Prompt:")
	fmt.Println(prompt)

	completion, err := generate(prompt)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println("Prediction:")
	fmt.Println(completion)
}
