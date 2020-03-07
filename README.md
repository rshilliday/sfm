# An implementation of incremental Structure from Motion

Structure from motion is an algorithm that generates a 3D reconstruction (pointcloud) from a sequence of 2D images. Instructions on running the repo can be found below. The rough steps of my pipeline are: 

i) Match keypoints between images  
ii) Find a good image pair to initialize the reconstruction (many matches and significant rotation between images)  
iii) Extend the reconstruction by resecting adjacent images with PnP and triangulating new points  
iv) Refine camera parameters and 3D point coordinates with bundle adjustment regularly

Here are examples of the output I was able to generate:

![](results/results_collage.png)


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
