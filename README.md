# An implementation of incremental Structure from Motion

If desired, clone this repository and setup the conda environment:
```
git clone https://github.com/rshilliday/sfm.git
cd sfm
conda env create -n sfm -f environment.yml
conda activate sfm
```

Now, run jupyter:
```
jupyter notebook
```
And open main.ipynb and hit "run all" to generate a 3D reconstruction of the "templeRing" dataset (http://vision.middlebury.edu/mview/data/).
I also created my own dataset from pictures of a Viking figurine. To generate a reconstruction of that, in the second cell of main.ipynb change n_imgs to 49 and change the imgset parameter of `find_features()` to 'Viking', and then hit "run all".
