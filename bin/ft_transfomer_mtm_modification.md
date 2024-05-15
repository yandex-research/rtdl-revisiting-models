# Write up on how to make the code in ft_transfomer to train unsupervised.

### Current model.
The current model is a transformer model that is trained on a supervised task. 
The model is trained to predict the item in category of classification or can also predict single value for regression task based on dataset.
The model is trained to minimize the cross-entropy loss or RMSE loss based on the config i.e the dataset it is used to train on.
Hence this model is only an encoder model that outputs of specific given dimensions.

Current model layers for given config:
```commandline
Transformer(
  (tokenizer): Tokenizer(
    (category_embeddings): Embedding(6, 64)
  )
  (layers): ModuleList(
    (0): ModuleDict(
      (attention): MultiheadAttention(
        (W_q): Linear(in_features=64, out_features=64, bias=True)
        (W_k): Linear(in_features=64, out_features=64, bias=True)
        (W_v): Linear(in_features=64, out_features=64, bias=True)
        (W_out): Linear(in_features=64, out_features=64, bias=True)
        (dropout): Dropout(p=0.5, inplace=False)
      )
      (linear0): Linear(in_features=64, out_features=128, bias=True)
      (linear1): Linear(in_features=64, out_features=64, bias=True)
      (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
    )
    (1): ModuleDict(
      (attention): MultiheadAttention(
        (W_q): Linear(in_features=64, out_features=64, bias=True)
        (W_k): Linear(in_features=64, out_features=64, bias=True)
        (W_v): Linear(in_features=64, out_features=64, bias=True)
        (W_out): Linear(in_features=64, out_features=64, bias=True)
        (dropout): Dropout(p=0.5, inplace=False)
      )
      (linear0): Linear(in_features=64, out_features=128, bias=True)
      (linear1): Linear(in_features=64, out_features=64, bias=True)
      (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (norm0): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
    )
  )
  (last_normalization): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
  (head): Linear(in_features=64, out_features=16, bias=True)
)
```
One can see that the model from class Transformer() at line 151 outputs 
a tensor of shape (batch_size, 16) which is then used to calculate the loss, assuming here 16 is vocab size.


### Modification to the model.
To make the model train in an unsupervised fashion we need to attach a decoder block to the base model.
This decoder block will attempt to predict the masked input tokens.
The model will be trained to minimize the cross-entropy loss only.

Please see the class TransformerEncoderDecoderModel() in line 326 of ft_transformer.py for the implementation
of how to build a encoder-decoder model using the give base transformer model.

### TODO
- Implement masking of input tokens at random at training time.
- Test the training loop for the unsupervised task.
- Test the evaluation loop for the unsupervised task.
- Test the inference loop for the unsupervised task.
- Test the model on a downstream task by fine-tuning it for a specific task.
