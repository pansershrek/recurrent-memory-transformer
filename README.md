### This branch is for replicating the experiments for the AAAI-24 paper: Beyond Attention: Breaking the Limits of Transformer Context Length with Recurrent Memory

RMT is a memory-augmented segment-level recurrent Transformer. It achieves state-of-the art results on Hyperpartisan dataset and beats Transformer-XL on algorithmic tasks and LM with limited input and memory size.

ToDo:

* [ ] Add code for algorithmic experiments

* [ ] Add code for LM experiments

* [ ] Add datasets

* [ ] Add links for papers


## Citation
If you find our work useful, please cite the RMT papers:
```
@inproceedings{
        bulatov2022recurrent,
        title={Recurrent Memory Transformer},
        author={Aydar Bulatov and Yuri Kuratov and Mikhail Burtsev},
        booktitle={Advances in Neural Information Processing Systems},
        editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
        year={2022},
        url={https://openreview.net/forum?id=Uynr3iPhksa}
}
```
```
@misc{bulatov2023scaling,
      title={Scaling Transformer to 1M tokens and beyond with RMT}, 
      author={Aydar Bulatov and Yuri Kuratov and Mikhail S. Burtsev},
      year={2023},
      eprint={2304.11062},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```