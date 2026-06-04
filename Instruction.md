Activate the conda env "/research/data/transfer/data/n41/fast_v2/v2" for any python execution.

Always follow this no matter what :Kepp a tack om implemntation details for each optimization, issues and fixes in CLAUDE.md


Analyze the batch_sample_method in generation_functions.py
I want to imnplement another method that does teh following:

Currently in batch sample we use eager mode execution with dynamic length KV caches and for every diffusion steps the KV for teh current block is concatenated creating more memory ops.We need to implementa a method named batch_sample_dyname, in  this we use statsic KV tensors based on max sequence length and allow support for torch comiple/dynamo/cuda graphs and eager mode for the method batch_sample_dynamo.

Along with this, I also want the feature to enable the mutation of input tensors like the huge static KV tensors so we can do in-place write for the current block's KV, without causing graph breaks. This might need settip up helper fucntions so create a utils directoty and keep teh helper functions and files there, if needed.

We also need a batch bucketting strategy, sop we keep the batch size fixed even if some of the request are done or completed, basically keep the batch size stastic so it doesn't casue grpah breaks and re-recordings.

We also need a hierarchy of attention backends as we want to pass the seqused variable to correctly commpute attention on the atual sequence length. We needa flaback mechanism that if flash-attention 4 cannot work then use flash--attention3 if not then use flash-attention 2, this can be based on the architecture of teh HPU so you might need to chek the web for flash-attention version details. 

The key goal here is to keep the number of cuda graphs to minimum and avoid in graph breaaks while not hurting the computation pipeline in the process.
Secondly, we are also trying to manage KV tensor better so, one optimisation is in-place write of KV tensors.
Any new optimization technique is also welcomed and appreciated, but let me know beofe implementing the idea.

Go ahead and implement this, make sure to keep the existing implementation intact, and use environment variables for the select different method or modes inside a method or any parameter for dynamo management.





Block Diffusion Model:

Okay, so first lets look how block diffusion works:
Firstly we havea  prefill phase where the promt goes through 1 forward steps and the KV is cached. Then succesive each block goes through diffusion step, and in this process all the tokens of that block go through bi-directional attention. The fisrt pass is prefill, where update the kv cache after the pass. In later denoising steps, eack block only cache KV at the end of diffusion so the intermeddiate steps create KV for a block and concatenate with existing KV for past blocks and then compute attention. 

We incorporate static KV tensor buffer, so each diffusion step should do an in-place write of generted KV for a block instead of torch.cat, this can be seen in the forward methods of /research/data/transfer/data/navneet/fast_v2/models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/modeling.py. For the batch_sample_dynamo method we need sepearet method in modelling.py named forward dynamo. This method will do a in-place write of current blocks KV tensors instead of torch.cat. This is true for every step with update_past_key_values=False, but for func call with past_key_values=True, it runs "past_key_value.update", so figure out how cn we do this with static cache in place, it has to be done while computing atttention returning teh KV tensor could increase memory operations.
Go ahead and implement this. 