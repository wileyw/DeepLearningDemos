from __future__ import print_function
import os

import numpy as np

import drawing
from data_frame import DataFrame
from drawing import alphabet

import svgwrite


class DataReader(object):

    def __init__(self, data_dir):
        data_cols = ['x', 'x_len', 'c', 'c_len']
        data = [np.load(os.path.join(data_dir, '{}.npy'.format(i))) for i in data_cols]

        self.test_df = DataFrame(columns=data_cols, data=data)
        self.train_df, self.val_df = self.test_df.train_test_split(train_size=0.95, random_state=2018)

        print('train size', len(self.train_df))
        print('val size', len(self.val_df))
        print('test size', len(self.test_df))

    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=True,
            num_epochs=10000,
            mode='train'
        )

    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=True,
            num_epochs=10000,
            mode='val'
        )

    def test_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.test_df,
            shuffle=False,
            num_epochs=1,
            mode='test'
        )

    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=10000, mode='train'):
        gen = df.batch_generator(
            batch_size=batch_size,
            shuffle=shuffle,
            num_epochs=num_epochs,
            allow_smaller_final_batch=(mode == 'test')
        )
        for batch in gen:
            batch['x_len'] = batch['x_len'] - 1
            max_x_len = np.max(batch['x_len'])
            max_c_len = np.max(batch['c_len'])
            batch['y'] = batch['x'][:, 1:max_x_len + 1, :]
            batch['x'] = batch['x'][:, :max_x_len, :]
            batch['c'] = batch['c'][:, :max_c_len]
            yield batch


def _draw(strokes, lines, filename, stroke_colors=None, stroke_widths=None):
    stroke_colors = stroke_colors or ['black']*len(lines)
    stroke_widths = stroke_widths or [2]*len(lines)

    line_height = 60
    view_width = 1000
    view_height = line_height*(len(strokes) + 1)

    dwg = svgwrite.Drawing(filename=filename)
    dwg.viewbox(width=view_width, height=view_height)
    dwg.add(dwg.rect(insert=(0, 0), size=(view_width, view_height), fill='white'))

    initial_coord = np.array([0, -(3*line_height / 4)])
    for offsets, line, color, width in zip(strokes, lines, stroke_colors, stroke_widths):

        if not line:
            initial_coord[1] -= line_height
            continue

        offsets[:, :2] *= 1.5
        strokes = drawing.offsets_to_coords(offsets)
        strokes = drawing.denoise(strokes)
        strokes[:, :2] = drawing.align(strokes[:, :2])

        strokes[:, 1] *= -1
        strokes[:, :2] -= strokes[:, :2].min() + initial_coord
        strokes[:, 0] += (view_width - strokes[:, 0].max()) / 2

        prev_eos = 1.0
        p = "M{},{} ".format(0, 0)
        for x, y, eos in zip(*strokes.T):
            p += '{}{},{} '.format('M' if prev_eos == 1.0 else 'L', x, y)
            prev_eos = eos
        path = svgwrite.path.Path(p)
        path = path.stroke(color=color, width=width, linecap='round').fill("none")
        dwg.add(path)

        initial_coord[1] -= line_height

    dwg.save()


def num_to_string(c, c_len):
    indices = c[:c_len - 1]
    str_out = ''.join([alphabet[x] for x in indices])
    return str_out


if __name__ == '__main__':
    dr = DataReader(data_dir='data/processed/')
    # import ipdb; ipdb.set_trace()

    stroke_colors = ['red', 'green', 'black', 'blue']
    stroke_widths = [1, 2, 1, 2]

    lines = [
        num_to_string(dr.test_df['c'][0], dr.test_df['c_len'][0]),
        num_to_string(dr.test_df['c'][1], dr.test_df['c_len'][1]),
    ]
    strokes = [
        dr.test_df['x'][0][:dr.test_df['x_len'][0]],
        dr.test_df['x'][1][:dr.test_df['x_len'][1]],
    ]

    import ipdb; ipdb.set_trace()

    _draw(strokes, lines, "test.svg", stroke_colors=stroke_colors, stroke_widths=stroke_widths)
