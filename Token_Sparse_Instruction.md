Activate the conda env "/research/data/transfer/data/navneet/fast_v2/v2" for any python execution.

Always follow this no matter what :Kepp a tack om implemntation details for each optimization, issues and fixes in CLAUDE.md

We need to build a new method named "batch_sample_sparse" almost exactly same as batch_sample_dynamo, but there is a small change which i described below.

So we run the denoising in two modes, dense and sparse. The first step will always be dense and we cache that blocks K and V in the static cache, along with this we also need a seperate block_size static cahche to cache the hidden state of each layer, lets call it teh block_sparse_cache. this buffer can be reset and reused accross blocks, once the current block is finish denoising, it can be reset for the next block.

We then use this cached hidden state to compare the next steps hidden state cosine similarity, which will give us and idea that which token is changing a lot and hence we will select a fraction of hidden state with lowest cosine similarity for recomputation through the decode layer. We will call this fracttion transfer ratio.  The first time in the dense step of a block, the decoder layer gets the hidden state it will cache it before passing it to input layernorm, and then in sucssesive sparse steps it will use teh cached hidden state to compute teh cosine similarity with current hidden statem, which will give us teh desired fraction of tokens with least cosine similarity, once we have this we can then cache the current hidde state. 

Once we get the desired token, we compute query, key, value for only these tokens and then we will update the static KV position for only these tokens, before computing attention. After attention we need to pass it thorugh similar steps of decoder layer including post attention layernorm and mlp, refer to the decoder layer of modelling.py, also keep in mind to create a seperate forward method in modelling.py if needed, don't touch the existing forward.

 It is essential to make sure that the static KV tensor hold the KV of current block in the right slot in the satic buffer accross steps, as this was not necessary for batch_sample or batch_sample_dynamo, but now this cached KV for the current block plays a huge role.

 We also need to cache the attn_output and mlp output, and after calculation of the few selected tokens we will scatter them in the cached attn_output tensor beofer post layernorm, and then simiklarly we will gather the same token selected based on cosine similarity to compute mlp and then scatter then again in the cached mlp output tensor before further processing.

 Cacahing strategy for attn_output and mlp_output will be same as hidden states, reuse and reset after block is finallized

 Also, we will have a parameter called refresh interval, this will determine that how frequently we need to call dense operation for teh same block, this means after how many step will we call the dense mode which will act on all block_size number of tokens and update all the caches.

 Go ahead and implement this, add env variable to parameters like tranfer ratio, refresh_interval and methods selection.


