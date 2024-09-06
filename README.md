![{4D84CC2F-D660-44F0-A5A1-E4EDA3349656}](https://github.com/user-attachments/assets/5f0d0c6c-f695-4c43-8dc2-6e6d9e0597fc)

 
# POS-Infused-Transformers-for-Machine-Translation.

This implimentation of a NAT attempted to alleviate the multimodality problem which occurs during machine translation with non-autoregressive transformers. 

The proposed approach iterated upon the works of Yang et al. (2021), Perera et al. (2022) and Li et al. (2019) by using part-of-speech tags to constrain non-autoregressive transformer output and guide the positional encoder. The positional encoder is guided by adding part-of-speech tags to the positional encodings in the encoder and decoder module. Furthermore, a “soft” constraint was proposed to be used in the architecture of Yang et al. (2021). Experiments were conducted on both the German-English dataset and the low-resource English-Vietnamese dataset. 

The obtained results contradict the works of Yang et al. (2021) as their architecture using a hard constraint is outperformed by the base CMLM architecture on the German-English dataset. Furthermore, our implementation of adding part-of-speech embeddings to the positional encoding results in the failure of the model to translate sentences in all experiments. Finally, the proposed soft constraints yield a small improvement over the base architecture on the low-resource language experiment. Using the soft constraint performed significantly better than the version proposed by Yang et al. (2021)

