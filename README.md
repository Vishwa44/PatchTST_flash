# PatchTST Supervised with flashattention

Objective:
Our project is to improve the PatchTST model, this model is a Transformer model for time series forecasting, with Flash Attention, a novel technique to improve the performance of the read/write operations that take place during Attention, which is projected to have a 3x improvement in the attention process, drastically improving the performance of the model.


Milestones:
  Baseline Profiling:
    Evaluate current model performance and resource usage.Document architecture, metrics, and resources. Flash Attention
  Implementation:
    Research Flash Attention mechanism. Implement in Python, test with sample data, and document.
  Integration with Current Model:
    Modify model to include Flash Attention. Retrain, evaluate, and document changes.
  Further Optimization:
    Analyze data characteristics. Experiment with techniques like preprocessing, optimization algorithms, and hardware utilization. Document optimization strategies and insights gained.  
  Developing a custom triton kernel:
    To reduce computation overhead for the smaller model size of PatchTST we developed a custom kernel which reads attention scores directly from froward pass which utilized in backward pass.

The custom kerenl implmentation is in a py notebook format in the Kernel folder.

Results:
  Our kernel has a 1.64 speed up for backward pass and 0.70 speed down for forward pass. 
  This kernel specifically performs better on smaller context length models.
  Due to the limitations of the library forward pass is written in an unoptimal way.
