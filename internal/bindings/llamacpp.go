package llamacppbindings

/*
#cgo CFLAGS: -std=c11
#cgo CXXFLAGS: -std=c++17

// === HEADER PATHS ===
// Default: From llama.cpp submodule (works for both source and binary builds)
// #cgo CPPFLAGS: -I${SRCDIR}/../../llama.cpp/ggml/include
// #cgo CPPFLAGS: -I${SRCDIR}/../../llama.cpp/include

// Alternative: From binary download (if you downloaded headers with Makefile.llama-binaries)
// Uncomment these two lines and comment out the two lines above to use binary headers:
// NOTE: Include order matters - include/ must come before include/ggml/
#cgo CPPFLAGS: -I${SRCDIR}/../../build/llama-binaries/include
#cgo CPPFLAGS: -I${SRCDIR}/../../build/llama-binaries/include/ggml

// === LIBRARY PATHS ===
// Default: Source build paths (build/llama.cpp)
// #cgo LDFLAGS: -L${SRCDIR}/../../build/llama.cpp/common
// #cgo LDFLAGS: -L${SRCDIR}/../../build/llama.cpp/src
// #cgo LDFLAGS: -L${SRCDIR}/../../build/llama.cpp/ggml/src

// Alternative: For binary builds, uncomment the line below and comment out the 3 lines above
#cgo LDFLAGS: -L${SRCDIR}/../../build/llama-binaries/lib

// === LINK LIBRARIES ===
// Updated for modern llama.cpp structure (ggml-base, ggml-cpu, etc.)
// Note: Binary downloads on Windows only provide .dll files
//       Run 'make prepare' to generate .dll.a import libraries
#cgo windows LDFLAGS: -l:libllama.dll.a -l:libggml.dll.a -l:libggml-base.dll.a -l:libggml-cpu.dll.a
#cgo linux LDFLAGS: -lllama -lggml -lggml-base -lggml-cpu
#cgo darwin LDFLAGS: -lllama -lggml -lggml-base -lggml-cpu
#cgo LDFLAGS: -lm -lstdc++
#cgo linux LDFLAGS: -lgomp
#cgo darwin LDFLAGS: -lomp

#include <stdlib.h>
#include "llama.h"
#include "gguf.h"
#include "ggml-backend.h"
extern bool llamaProgressCallback(float progress, void *user_data);
extern void llamaLog(int level, char* text, void* user_data);
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"runtime/cgo"
	"strings"
	"unsafe"

	"github.com/hypernetix/llamacpp_server/internal/logging"
)

func init() {
	// Load all available GGML backends (CPU, CUDA, Metal, etc.)
	// This is required in modern llama.cpp versions before loading models
	C.ggml_backend_load_all()

	// Set up logging callback
	C.llama_log_set(C.ggml_log_callback(C.llamaLog), nil)
}

var globalLogger logging.SprintfLogger

//export llamaLog
func llamaLog(level C.int, text *C.char, _ unsafe.Pointer) {
	if globalLogger == nil {
		return
	}
	switch int(level) {
	case C.GGML_LOG_LEVEL_DEBUG:
		globalLogger.Debugf(C.GoString(text))
	case C.GGML_LOG_LEVEL_INFO:
		globalLogger.Infof(C.GoString(text))
	case C.GGML_LOG_LEVEL_ERROR:
		globalLogger.Errorf(C.GoString(text))
	case C.GGML_LOG_LEVEL_WARN:
		globalLogger.Warnf(C.GoString(text))
	}
}

//export llamaProgressCallback
func llamaProgressCallback(progress C.float, userData unsafe.Pointer) C.bool {
	handle := *(*cgo.Handle)(userData)
	callback := handle.Value().(func(float32))
	callback(float32(progress))
	return true
}

func Initialize(logger logging.SprintfLogger) {
	globalLogger = logger
	C.ggml_backend_load_all()
	C.llama_backend_init()
}

func GetModelArch(modelPath string) (string, error) {
	mp := C.CString(modelPath)
	defer C.free(unsafe.Pointer(mp))

	gguf_ctx := C.gguf_init_from_file(mp, C.struct_gguf_init_params{no_alloc: true, ctx: (**C.struct_ggml_context)(C.NULL)})
	if gguf_ctx == nil {
		return "", errors.New("unable to load model file")
	}
	defer C.gguf_free(gguf_ctx)

	key := C.CString("general.architecture")
	defer C.free(unsafe.Pointer(key))
	arch_index := C.gguf_find_key(gguf_ctx, key)
	if int(arch_index) < 0 {
		return "", errors.New("unknown model architecture")
	}

	arch := C.gguf_get_val_str(gguf_ctx, arch_index)

	return C.GoString(arch), nil
}

type ModelParams struct {
	impl              C.struct_llama_model_params
	progressHandlePin *runtime.Pinner
	tensorSplitPin    *runtime.Pinner
}

func NewModelDefaultParams() *ModelParams {
	impl := C.llama_model_default_params()
	return &ModelParams{impl: impl}
}

func (p *ModelParams) SetNGpuLayers(nGpuLayers int) {
	p.impl.n_gpu_layers = C.int(nGpuLayers)
}

func (p *ModelParams) SetMainGpu(mainGpu int) {
	p.impl.main_gpu = C.int32_t(mainGpu)
}

func (p *ModelParams) SetUseMmap(useMmap bool) {
	p.impl.use_mmap = C.bool(useMmap)
}

func (p *ModelParams) SetUseMlock(useMlock bool) {
	p.impl.use_mlock = C.bool(useMlock)
}

func (p *ModelParams) SetVocabOnly(vocabOnly bool) {
	p.impl.vocab_only = C.bool(vocabOnly)
}

func (p *ModelParams) SetTensorSplit(tensorSplit []float32) {
	if len(tensorSplit) == 0 {
		p.freeTensorSplitPin()
		return
	}
	tensorSplitData := &tensorSplit[0]
	var tensorSplitPin runtime.Pinner
	tensorSplitPin.Pin(tensorSplitData)
	p.impl.tensor_split = (*C.float)(unsafe.Pointer(tensorSplitData))
	p.tensorSplitPin = &tensorSplitPin
}

func (p *ModelParams) SetProgressCallback(progress func(float32)) {
	if progress == nil {
		p.freeProgressHandle()
		p.impl.progress_callback = nil
		return
	}

	p.impl.progress_callback = C.llama_progress_callback(C.llamaProgressCallback)

	handle := cgo.NewHandle(progress)
	var handlePin runtime.Pinner
	handlePin.Pin(&handle)
	p.impl.progress_callback_user_data = unsafe.Pointer(&handle)
	p.progressHandlePin = &handlePin
}

func (p *ModelParams) freeTensorSplitPin() {
	p.impl.tensor_split = nil
	if p.tensorSplitPin != nil {
		p.tensorSplitPin.Unpin()
	}
	p.tensorSplitPin = nil
}

func (p *ModelParams) freeProgressHandle() {
	if p.progressHandlePin != nil {
		p.progressHandlePin.Unpin()
	}
	p.progressHandlePin = nil
	if p.impl.progress_callback_user_data != nil {
		handle := *(*cgo.Handle)(p.impl.progress_callback_user_data)
		handle.Delete()
	}
	p.impl.progress_callback_user_data = nil
}

func (p *ModelParams) Free() {
	p.freeProgressHandle()
	p.freeTensorSplitPin()
}

type ModelInfo struct {
	Desc        string
	Size        uint64
	NParams     uint64
	HasEncoder  bool
	HasDecoder  bool
	IsRecurrent bool
}

type Model struct {
	impl *C.struct_llama_model
}

func LoadModelFromFile(modelPath string, params *ModelParams) (*Model, error) {
	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	impl := C.llama_model_load_from_file(cModelPath, params.impl)
	if impl == nil {
		return nil, fmt.Errorf("unable to load model: %s", modelPath)
	}

	return &Model{impl: impl}, nil
}

func (m *Model) Free() {
	C.llama_model_free(m.impl)
}

func (m *Model) Info() ModelInfo {
	const bufLen = 1024
	buf := make([]byte, bufLen)
	descLen := int(C.llama_model_desc(m.impl, (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(bufLen)))
	info := ModelInfo{
		Desc:        string(buf[:descLen]),
		Size:        uint64(C.llama_model_size(m.impl)),
		NParams:     uint64(C.llama_model_n_params(m.impl)),
		HasEncoder:  bool(C.llama_model_has_encoder(m.impl)),
		HasDecoder:  bool(C.llama_model_has_decoder(m.impl)),
		IsRecurrent: bool(C.llama_model_is_recurrent(m.impl)),
	}
	return info
}

type Vocab struct {
	impl *C.struct_llama_vocab
}

type VocabInfo struct {
	Type int
}

func (m *Model) Vocab() *Vocab {
	return &Vocab{impl: C.llama_model_get_vocab(m.impl)}
}

func (v *Vocab) Info() VocabInfo {
	return VocabInfo{
		Type: int(C.llama_vocab_type(v.impl)),
	}
}

func (v *Vocab) NTokens() int {
	return int(C.llama_vocab_n_tokens(v.impl))
}

func (v *Vocab) AddBOS() bool {
	return bool(C.llama_vocab_get_add_bos(v.impl))
}

func (v *Vocab) IsEog(token int) bool {
	return bool(C.llama_vocab_is_eog(v.impl, C.llama_token(token)))
}

func (v *Vocab) Tokenize(text string, addSpecial bool, parseSpecial bool) ([]int, error) {
	maxTokens := len(text) + 2
	cTokens := make([]C.llama_token, maxTokens)
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.llama_tokenize(
		v.impl,
		cText,
		C.int32_t(len(text)),
		&cTokens[0],
		C.int32_t(maxTokens),
		C.bool(addSpecial),
		C.bool(parseSpecial),
	)

	// if the result is negative, reallocate and retry with the correct buffer size
	if result < 0 {
		maxTokens = int(-result)
		cTokens = make([]C.llama_token, maxTokens)
		result = C.llama_tokenize(
			v.impl,
			cText,
			C.int32_t(len(text)),
			&cTokens[0],
			C.int32_t(maxTokens),
			C.bool(addSpecial),
			C.bool(parseSpecial),
		)
		if result < 0 {
			return nil, fmt.Errorf("tokenization failed, required %d tokens", -result)
		}
	}

	tokens := make([]int, result)
	for i := range result {
		tokens[i] = int(cTokens[i])
	}

	return tokens, nil
}

func (v *Vocab) TokenToPiece(token int) (string, error) {
	bufLen := 256
	buf := make([]byte, bufLen)
	bufLen = int(C.llama_token_to_piece(
		v.impl,
		C.int32_t(token),
		(*C.char)(unsafe.Pointer(&buf[0])),
		C.int32_t(bufLen),
		C.int32_t(0),
		C.bool(true),
	))

	if bufLen < 0 {
		bufLen = -bufLen
		buf = make([]byte, bufLen)
		if C.llama_token_to_piece(
			v.impl,
			C.int32_t(token),
			(*C.char)(unsafe.Pointer(&buf[0])),
			C.int32_t(bufLen),
			C.int32_t(0),
			C.bool(true),
		) < 0 {
			return "", fmt.Errorf("token to piece failed")
		}
	}

	return strings.TrimRight(string(buf), "\x00"), nil
}

type ContextParams struct {
	impl C.struct_llama_context_params
}

func NewContextDefaultParams() *ContextParams {
	impl := C.llama_context_default_params()
	return &ContextParams{impl: impl}
}

func (p *ContextParams) NCtx() int {
	return int(p.impl.n_ctx)
}

func (p *ContextParams) NBatch() int {
	return int(p.impl.n_batch)
}

func (p *ContextParams) NSeqMax() int {
	return int(p.impl.n_seq_max)
}

func (p *ContextParams) NThreads() int {
	return int(p.impl.n_threads)
}

func (p *ContextParams) NThreadsBatch() int {
	return int(p.impl.n_threads_batch)
}

func (p *ContextParams) SetNCtx(nCtx int) {
	p.impl.n_ctx = C.uint32_t(nCtx)
}

func (p *ContextParams) SetNBatch(nBatch int) {
	p.impl.n_batch = C.uint32_t(nBatch)
}

func (p *ContextParams) SetNSeqMax(nSeqMax int) {
	p.impl.n_seq_max = C.uint32_t(nSeqMax)
}

func (p *ContextParams) SetNThreads(nThreads int) {
	p.impl.n_threads = C.int32_t(nThreads)
}

func (p *ContextParams) SetNThreadsBatch(nThreadsBatch int) {
	p.impl.n_threads_batch = C.int32_t(nThreadsBatch)
}

func (p *ContextParams) SetEmbeddings(embeddings bool) {
	p.impl.embeddings = C.bool(embeddings)
}

func (p *ContextParams) SetFlashAttention(flashAttention bool) {
	// Note: In llama.cpp b6770+, flash_attn changed to flash_attn_type (enum)
	// LLAMA_FLASH_ATTN_TYPE_DISABLED = 0, LLAMA_FLASH_ATTN_TYPE_ENABLED = 1
	if flashAttention {
		p.impl.flash_attn_type = C.LLAMA_FLASH_ATTN_TYPE_ENABLED
	} else {
		p.impl.flash_attn_type = C.LLAMA_FLASH_ATTN_TYPE_DISABLED
	}
}

func (p *ContextParams) SetTypeKV(kvCacheType string) {
	p.impl.type_k = kvCacheTypeFromStr(strings.ToLower(kvCacheType))
	p.impl.type_v = kvCacheTypeFromStr(strings.ToLower(kvCacheType))
}

func kvCacheTypeFromStr(s string) C.enum_ggml_type {
	if s == "" {
		return C.GGML_TYPE_F16
	}
	switch s {
	case "q8_0":
		return C.GGML_TYPE_Q8_0
	case "q4_0":
		return C.GGML_TYPE_Q4_0
	default:
		return C.GGML_TYPE_F16
	}
}

type Context struct {
	impl *C.struct_llama_context
}

func NewContext(model *Model, params *ContextParams) (*Context, error) {
	impl := C.llama_init_from_model(model.impl, params.impl)
	if impl == nil {
		return nil, fmt.Errorf("unable to create context")
	}
	return &Context{impl: impl}, nil
}

func (c *Context) Free() {
	C.llama_free(c.impl)
}

func (c *Context) NCells() int {
	return int(C.llama_n_ctx(c.impl))
}

// NCellsUsed returns an estimate of the number of used cells in the KV cache.
// Note: In llama.cpp b6770+, the old llama_kv_self_used_cells() function was removed.
// This implementation uses the new memory API to estimate usage based on sequence 0.
// For a more accurate count across all sequences, use the memory API directly.
func (c *Context) NCellsUsed() int {
	// Get the memory handle
	mem := C.llama_get_memory(c.impl)
	if mem == nil {
		return 0
	}

	// Get the max position for sequence 0 (the primary sequence)
	// This gives us an approximation of used cells for single-sequence inference
	maxPos := int(C.llama_memory_seq_pos_max(mem, 0))

	// If sequence is empty, max_pos will be -1
	if maxPos < 0 {
		return 0
	}

	// The number of used cells is approximately max_pos + 1
	// (positions are 0-indexed, so position 5 means 6 cells used: 0,1,2,3,4,5)
	return maxPos + 1
}

var ErrKvCacheFull = errors.New("could not find a kv cache slot")

func (c *Context) Decode(batch *Batch) error {
	// Positive return values does not mean a fatal error, but rather a warning.
	//   0 - success
	//   1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
	// < 0 - error
	result := int(C.llama_decode(c.impl, batch.impl))

	if result < 0 {
		return fmt.Errorf("failed to decode: %d", result)
	}

	if result > 0 {
		return ErrKvCacheFull
	}

	return nil
}

type Sampler struct {
	impl *C.struct_llama_sampler
}

func (s *Sampler) Free() {
	C.llama_sampler_free(s.impl)
}

func NewMinPSampler(minP float32, minKeep int) (*Sampler, error) {
	impl := C.llama_sampler_init_min_p(C.float(minP), C.size_t(minKeep))
	if impl == nil {
		return nil, errors.New("unable to create min_p sampler")
	}
	return &Sampler{impl: impl}, nil
}

func NewTempSampler(temp float32) (*Sampler, error) {
	impl := C.llama_sampler_init_temp(C.float(temp))
	if impl == nil {
		return nil, errors.New("unable to create temp sampler")
	}
	return &Sampler{impl: impl}, nil
}

// NewTopKSampler creates a top-k sampler
func NewTopKSampler(k int) (*Sampler, error) {
	impl := C.llama_sampler_init_top_k(C.int32_t(k))
	if impl == nil {
		return nil, errors.New("unable to create top_k sampler")
	}
	return &Sampler{impl: impl}, nil
}

// NewTopPSampler creates a top-p (nucleus) sampler
func NewTopPSampler(p float32, minKeep int) (*Sampler, error) {
	impl := C.llama_sampler_init_top_p(C.float(p), C.size_t(minKeep))
	if impl == nil {
		return nil, errors.New("unable to create top_p sampler")
	}
	return &Sampler{impl: impl}, nil
}

// NewPenaltiesSampler creates a penalties sampler for repetition control
func NewPenaltiesSampler(penaltyLastN int, penaltyRepeat, penaltyFreq, penaltyPresent float32) (*Sampler, error) {
	impl := C.llama_sampler_init_penalties(
		C.int32_t(penaltyLastN),
		C.float(penaltyRepeat),
		C.float(penaltyFreq),
		C.float(penaltyPresent),
	)
	if impl == nil {
		return nil, errors.New("unable to create penalties sampler")
	}
	return &Sampler{impl: impl}, nil
}

// NewDistSampler creates a distribution sampler for final token selection
func NewDistSampler(seed uint32) (*Sampler, error) {
	impl := C.llama_sampler_init_dist(C.uint32_t(seed))
	if impl == nil {
		return nil, errors.New("unable to create dist sampler")
	}
	return &Sampler{impl: impl}, nil
}

// SafeNewTopKSampler creates a top-k sampler with fallback
func SafeNewTopKSampler(k int) (*Sampler, error) {
	// Try to create top-k sampler, fallback to greedy if not available
	defer func() {
		if r := recover(); r != nil {
			// Function not available in this llama.cpp build
		}
	}()

	impl := C.llama_sampler_init_top_k(C.int32_t(k))
	if impl == nil {
		return nil, errors.New("unable to create top_k sampler")
	}
	return &Sampler{impl: impl}, nil
}

// SafeNewTopPSampler creates a top-p sampler with fallback
func SafeNewTopPSampler(p float32, minKeep int) (*Sampler, error) {
	defer func() {
		if r := recover(); r != nil {
			// Function not available in this llama.cpp build
		}
	}()

	impl := C.llama_sampler_init_top_p(C.float(p), C.size_t(minKeep))
	if impl == nil {
		return nil, errors.New("unable to create top_p sampler")
	}
	return &Sampler{impl: impl}, nil
}

// SafeNewPenaltiesSampler creates a penalties sampler with fallback
func SafeNewPenaltiesSampler(penaltyLastN int, penaltyRepeat, penaltyFreq, penaltyPresent float32) (*Sampler, error) {
	defer func() {
		if r := recover(); r != nil {
			// Function not available in this llama.cpp build
		}
	}()

	impl := C.llama_sampler_init_penalties(
		C.int32_t(penaltyLastN),
		C.float(penaltyRepeat),
		C.float(penaltyFreq),
		C.float(penaltyPresent),
	)
	if impl == nil {
		return nil, errors.New("unable to create penalties sampler")
	}
	return &Sampler{impl: impl}, nil
}

func NewSeedSampler(seed uint32) (*Sampler, error) {
	impl := C.llama_sampler_init_dist(C.uint32_t(seed))
	if impl == nil {
		return nil, fmt.Errorf("unable to create seed sampler")
	}
	return &Sampler{impl: impl}, nil
}

func NewGreedySampler() (*Sampler, error) {
	impl := C.llama_sampler_init_greedy()
	if impl == nil {
		return nil, fmt.Errorf("unable to create greedy sampler")
	}
	return &Sampler{impl: impl}, nil
}

func NewGrammarSampler(vocab *Vocab, grammar string) (*Sampler, error) {
	cGrammar := C.CString(grammar)
	cRoot := C.CString("root")
	defer C.free(unsafe.Pointer(cGrammar))
	defer C.free(unsafe.Pointer(cRoot))
	impl := C.llama_sampler_init_grammar(vocab.impl, cGrammar, cRoot)
	if impl == nil {
		return nil, fmt.Errorf("unable to create grammar sampler")
	}
	return &Sampler{impl: impl}, nil
}

func (s *Sampler) Sample(context *Context, idx int) int {
	return int(C.llama_sampler_sample(s.impl, context.impl, C.int32_t(idx)))
}

type SamplerChainParams struct {
	impl C.struct_llama_sampler_chain_params
}

func NewSamplerChainDefaultParams() *SamplerChainParams {
	impl := C.llama_sampler_chain_default_params()
	return &SamplerChainParams{impl: impl}
}

type SamplerChain struct {
	impl *C.struct_llama_sampler
}

func NewSamplerChain(params *SamplerChainParams) (*SamplerChain, error) {
	impl := C.llama_sampler_chain_init(params.impl)
	if impl == nil {
		return nil, fmt.Errorf("unable to create sampler chain")
	}
	return &SamplerChain{impl: impl}, nil
}

func (s *SamplerChain) AddSampler(sampler *Sampler) {
	C.llama_sampler_chain_add(s.impl, sampler.impl)
}

func (s *SamplerChain) Sampler() *Sampler {
	return &Sampler{impl: s.impl}
}

func (s *SamplerChain) Free() {
	C.llama_sampler_free(s.impl)
}

type Batch struct {
	owned bool
	impl  C.struct_llama_batch
}

func NotOwnedOneItemBatch(tokens []int) *Batch {
	cTokens := make([]C.llama_token, len(tokens))
	for i := range tokens {
		cTokens[i] = C.llama_token(tokens[i])
	}
	impl := C.llama_batch_get_one(&cTokens[0], C.int32_t(len(tokens)))
	return &Batch{owned: false, impl: impl} // will not be freed
}

func (b *Batch) Free() {
	if b.owned {
		C.llama_batch_free(b.impl)
	}
}

func (b *Batch) NTokens() int {
	return int(b.impl.n_tokens)
}
