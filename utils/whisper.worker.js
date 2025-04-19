import { pipeline } from '@xenova/transformers';
import { MessageTypes } from './presets';

class MyTranscriptionPipeline {
    static task = 'automatic-speech-recognition';
    static model = 'openai/whisper-tiny.en';
    static instance = null;

    static async getInstance(progress_callback = null) {
        if (this.instance === null) {
            try {
                this.instance = await pipeline(this.task, MyTranscriptionPipeline.model, {
                    progress_callback,
                });
            } catch (err) {
                console.error('Pipeline load error:', err);
                throw err;
            }
        }
        return this.instance;
    }
}

self.addEventListener('message', async (event) => {
    const { type, audio } = event.data;
    if (type === MessageTypes.INFERENCE_REQUEST) {
        await transcribe(audio);
    }
});

async function transcribe(audio) {
    sendLoadingMessage('loading');

    let pipelineInstance;

    try {
        pipelineInstance = await MyTranscriptionPipeline.getInstance(load_model_callback);
    } catch (err) {
        console.error('Failed to load pipeline:', err.message);
        sendLoadingMessage('error');
        return;
    }

    sendLoadingMessage('success');

    const stride_length_s = 5;
    const generationTracker = new GenerationTracker(pipelineInstance, stride_length_s);

    await pipelineInstance(audio, {
        top_k: 0,
        do_sample: false,
        chunk_length: 30,
        stride_length_s,
        return_timestamps: true,
        callback_function: generationTracker.callbackFunction.bind(generationTracker),
        chunk_callback: generationTracker.chunkCallback.bind(generationTracker),
    });

    generationTracker.sendFinalResult();
}

async function load_model_callback(data) {
    if (data.status === 'progress') {
        const { file, progress, loaded, total } = data;
        sendDownloadingMessage(file, progress, loaded, total);
    }
}

function sendLoadingMessage(status) {
    self.postMessage({
        type: MessageTypes.LOADING,
        status,
    });
}

async function sendDownloadingMessage(file, progress, loaded, total) {
    self.postMessage({
        type: MessageTypes.DOWNLOADING,
        file,
        progress,
        loaded,
        total,
    });
}

class GenerationTracker {
    constructor(pipeline, stride_length_s) {
        this.pipeline = pipeline;
        this.stride_length_s = stride_length_s;
        this.chunks = [];
        this.processed_chunks = [];
        this.callbackFunctionCounter = 0;

        const processor = pipeline?.processor;
        const model = pipeline?.model;

        this.time_precision =
            processor?.feature_extractor?.config?.chunk_length /
            model?.config?.max_source_positions || 0.02; // fallback
    }

    sendFinalResult() {
        self.postMessage({ type: MessageTypes.INFERENCE_DONE });
    }

    callbackFunction(beams) {
        this.callbackFunctionCounter++;
        if (this.callbackFunctionCounter % 10 !== 0) return;

        const bestBeam = beams[0];
        let text = this.pipeline.tokenizer.decode(bestBeam.output_token_ids, {
            skip_special_tokens: true,
        });

        const result = {
            text,
            start: this.getLastChunkTimestamp(),
            end: undefined,
        };

        createPartialResultMessage(result);
    }

    chunkCallback(data) {
        this.chunks.push(data);
        const [text, { chunks }] = this.pipeline.tokenizer._decode_asr(this.chunks, {
            time_precision: this.time_precision,
            return_timestamps: true,
            force_full_sequence: false,
        });

        this.processed_chunks = chunks.map((chunk, index) => this.processChunk(chunk, index));
        createResultMessage(this.processed_chunks, false, this.getLastChunkTimestamp());
    }

    getLastChunkTimestamp() {
        if (this.processed_chunks.length === 0) return 0;
        const lastChunk = this.processed_chunks[this.processed_chunks.length - 1];
        return lastChunk.end;
    }

    processChunk(chunk, index) {
        const { text, timestamp } = chunk;
        const [start, end] = timestamp;

        return {
            index,
            text: text.trim(),
            start: Math.round(start),
            end: Math.round(end) || Math.round(start + 0.9 * this.stride_length_s),
        };
    }
}

function createResultMessage(results, isDone, completedUntilTimestamp) {
    self.postMessage({
        type: MessageTypes.RESULT,
        results,
        isDone,
        completedUntilTimestamp,
    });
}

function createPartialResultMessage(result) {
    self.postMessage({
        type: MessageTypes.RESULT_PARTIAL,
        result,
    });
}
