3D brain segmentation
---------------------

The coherent 3D volume generated by our pipeline can be used for downstream analysis, 
for example, segmentation or parcellation of 3D brain volume. 
The ``segment-volume`` command provides a wrapper of the 
`TWAI segmentation algorithm <https://github.com/LucasFidon/trustworthy-ai-fetal-brain-segmentation>`_ 
for T2w fetal brain MRI. An exmaple of ``segment-volume`` is as follows:

.. code-block:: nesvorcommand

    nesvor segment-volume
        --input-volume reconstructed-volume.nii.gz \
        --output-folder <path-to-save-segmentation>

If the gestational age and condition (i.e., Neurotypical, Spina Bifida, or Pathological) of the subject are known, they 
can be specified in the command.

.. code-block:: nesvorcommand

    nesvor segment-volume
        --input-volume reconstructed-volume.nii.gz \
        --output-folder <path-to-save-segmentation> \
        --ga 32 \
        --condition Neurotypical