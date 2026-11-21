import unittest
import numpy as np

from bop_bb8.yolo_photometric import YOLOPhotometric


class TestYOLOPhotometric(unittest.TestCase):
    def test_boxes_and_source_unchanged(self):
        image=np.random.default_rng(8).integers(0,256,(64,96,3),dtype=np.uint8)
        source=image.copy();boxes=np.array([[10,12,54,46]],dtype=np.float32)
        labels=dict(img=image,instances=boxes,cls=np.array([[0]]),ratio_pad=(1.,1.))
        transform=YOLOPhotometric();np.random.seed(42);a=transform(labels)
        np.random.seed(42);b=transform(labels)
        np.testing.assert_array_equal(a['img'],b['img'])
        np.testing.assert_array_equal(source,image)
        self.assertEqual(a['img'].shape,image.shape)
        for k in labels:
            if k!='img':self.assertIs(a[k],labels[k])


if __name__=='__main__':unittest.main()
