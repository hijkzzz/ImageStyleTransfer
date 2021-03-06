# InstaGAN

## Summary
![here](https://github.com/SerialLain3170/Style-Transfer/blob/master/InstaGAN/newwork.png)

- I implement hair color style transfer using InstaGAN after preparing segmentation mask of extracting the region of hair.

## Usage

### Training Phase
Execute the command line below.

```
$ python train.py --src_path <SRC_PATH> --tgt_path <TGT_PATH>
```
`SRC_PATH` and `TGT_PATH` are direcotry which contains training images and masks of each domain.
The structure of directory is as follows.

```
SRC_PATH(TGT_PATH) - image - file1
                           - file2
                           - ...
                           
                   - mask  - file1
                           - file2
                           - ...
```

## Result
Result generated by my development environment is below.
![here](https://github.com/SerialLain3170/Style-Transfer/blob/master/InstaGAN/result.png)

The quality of conversion is insufficient, but it seems that cource of action is not wrong.

- Batch size: 3
- Using Adam as optimizer
- Using horizontal flip as data augmentation
