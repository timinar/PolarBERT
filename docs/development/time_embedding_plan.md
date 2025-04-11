
**Phase 1: Data Exploration & Preprocessing Definition**

1.  **Goal:** Understand the distributions of `time` and `charge` in the raw Kaggle data to inform preprocessing choices.
2.  **Tasks:**
    * Load a representative sample of events from one or more `batch_*.parquet` files.
    * Analyze the `time` feature:
        * Determine the typical range (min, max, mean, median).
        * Plot its distribution (histogram). Is it uniform? Skewed?
        * Define the transformation to integer: `time_int = int(round(time_float * 3e4 + 1e4))`. Verify the resulting integer range. Decide on the vocabulary size for the time embedding (e.g., `max(time_int) + 1`).
    * Analyze the `charge` feature:
        * Plot its distribution (histogram, potentially on a log scale).
        * Determine a quantization strategy (e.g., logarithmic binning, quantile binning, or simple linear binning). Decide on the number of bins (vocabulary size for charge embedding).
        * Define the transformation: `charge_bin = quantize(charge_float)`.
3.  **Outcome:** Clear definitions for `time_int` transformation, `charge_bin` quantization, and the vocabulary sizes needed for the embedding layers.

**Phase 2: Develop New Embedding Layer (`notebook`)**

1.  **Goal:** Create and test a new `nn.Module` for embedding IceCube pulse features.
2.  **Tasks:**
    * Implement the `time_int` and `charge_bin` transformations in Python/NumPy based on Phase 1 findings.
    * Create a new `nn.Module` (e.g., `EnhancedIceCubeEmbedding`).
        * Define `nn.Embedding` layers for:
            * `sensor_id` (similar to existing, `num_embeddings=5160 + 2`, remember padding/mask tokens).
            * `time_int` (using vocabulary size from Phase 1).
            * `charge_bin` (using vocabulary size from Phase 1).
        * Define a way to handle the `auxiliary` feature (e.g., a small `nn.Embedding` with 2-3 embeddings, or a simple linear projection if preferred).
        * Concatenate the embeddings from `sensor_id`, `time_int`, `charge_bin`, and `auxiliary`.
        * Optionally, add a final linear layer to project the concatenated embedding to the desired `model.embedding_dim`.
        * Ensure handling of padding and potential masking tokens.
        * Include the prepending of the CLS token.
    * Test the forward pass with dummy data matching the expected input structure.
3.  **Outcome:** A working `EnhancedIceCubeEmbedding` module tested in a notebook environment.

**Phase 3: Integrate Embedding into Full Model (`notebook` -> `script`)**

1.  **Goal:** Replace the old embedding layer in a transformer model architecture and verify the forward pass.
2.  **Tasks:**
    * Copy an existing model structure (e.g., from `flash_model.py`) into the notebook.
    * Replace the old `IceCubeEmbedding` instance with the new `EnhancedIceCubeEmbedding` instance.
    * Ensure the dimensions match between the embedding output and the transformer blocks.
    * Test the full model's forward pass with dummy data.
    * Once working in the notebook, create a new Python script (e.g., `src/polarbert/enhanced_embedding_model.py`) containing the new embedding and the model definition (e.g., `EnhancedFlashTransformer`).
3.  **Outcome:** A Python script defining the complete transformer model using the new embedding layer.

**Phase 4: Adapt Training Script**

1.  **Goal:** Modify the pre-training or fine-tuning script to use the new model and data transformations.
2.  **Tasks:**
    * Create a new training script (e.g., `scripts/pretrain_enhanced.py`) by copying and modifying `pretraining.py`.
    * **Crucially:** Modify the `get_dataloaders` function or the `IceCubeDataset` itself to apply the new `time_int` and `charge_bin` transformations *before* the data is passed to the model. This might involve adding new arguments to `get_dataloaders` or creating a new dataset class variant if the changes are substantial. *Avoid changing the memmap creation script for now.* Apply transformations on-the-fly in the dataloader.
    * Update the script to import and instantiate the new model (e.g., `EnhancedFlashTransformer`).
    * Adjust configuration files (`*.yaml`) to potentially include parameters for the new embedding (like quantization bins) or point to the new model type.
    * Run a minimal training loop locally (few steps) to ensure data loading and model forward/backward passes work without crashing.
3.  **Outcome:** A runnable training script capable of training the model with the enhanced embedding.

**Phase 5: Training & Evaluation**

1.  **Goal:** Train the new model and compare its performance to the baseline.
2.  **Tasks:**
    * Perform pre-training runs using the script from Phase 4. Monitor training/validation loss (especially DOM prediction and charge regression losses).
    * Perform fine-tuning runs (e.g., directional prediction) using the pre-trained model.
    * Compare the final metrics (e.g., validation loss, angular error for fine-tuning) against the original model baseline.
3.  **Outcome:** Performance comparison results to determine if the new embedding strategy is beneficial.

**Phase 6: (Optional Future Step) Add Positional Encoding (RoPE)**

1.  **Goal:** Further enhance the model by incorporating relative positional information.
2.  **Tasks:**
    * Integrate RoPE into the attention mechanism (e.g., `flash_model.py`'s `Attention` module). This usually involves modifying how Query and Key vectors are computed before the dot-product attention. Libraries like `rotary_embedding_torch` can simplify this.
    * Retrain and re-evaluate the model.
3.  **Outcome:** Assessment of whether RoPE provides additional performance gains on top of the enhanced time/charge embedding.