This is the project page of our paper:  

**Learning Neural Bag-of-Matrix-Summarization with Riemannian Network**,  
Liu, H., Li, J., Wu, Y., & Ji, R.
AAAI 2019.
[[PDF]](https://aaai.org/ojs/index.php/AAAI/article/view/4899/4772)

## Code
Check `main.py` for details.

Our codes are based on [zzhiwu/SPDNet](https://github.com/zzhiwu/SPDNet), and translated into PyTorch.

Different from the original paper, we find that replacing the classifier with the softmax on the distances between the dictionary is more stable.

Note that best of results on AFEW is fine-tuned from the SPDNet.

Feel free to contact to the authors (lijie.32@outlook.com) if you have any problems.

## Citation  
If our paper helps your research, please cite it in your publications:
```
@InProceedings{liu2019learning,
  title={Learning Neural Bag-of-Matrix-Summarization with Riemannian Network},
  author={Liu, Hong and Li, Jie and Wu, Yongjian and Ji, Rongrong},
  booktitle={Association for the Advancement of Artificial Intelligence (AAAI)},
  year={2019}
}

@InProceedings{huang2017riemannian,
  title={A riemannian network for spd matrix learning},
  author={Huang, Zhiwu and Van Gool, Luc},
  booktitle={Thirty-First AAAI Conference on Artificial Intelligence},
  year={2017}
}
```
