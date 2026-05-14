'''
Test parameter extraction from noise function
'''

import pandas as pd
import numpy as np
from clonesearch.utils import get_params as params

class TestMakeDf:
    '''
    Test output
    '''
    def test_make_output_df(self):

        N = {
            ('time1','time3'): {
                                ('75','50'):  30,
                                ('100','75'): 50
                            },
        }
        avgs = {
            ('time1','time3'): {
                                ('75','50'):  1,
                                ('100','75'): 5
                            },
        }
        dt = {
            ('time1','time3'): {
                                ('75','50'):  3,
                                ('100','75'): 2
                            },
        }
        f = {
            ('time1','time3'): {
                                ('75','50'):  4.5,
                                ('100','75'): 5.5
                            },
        }


        output_df = params._make_output_df(N, avgs, dt, f)

        expected_output = pd.DataFrame(
            columns = ['p1','p2','t1','t2', 'avgs', 'N', 'dt', 'freqs', 'percentiles'],
            data = [
                ['75', '50', 'time1', 'time3', 1, 30, 3, 4.5, '75-50'],
                ['100', '75', 'time1', 'time3', 5, 50, 2, 5.5, '100-75']
            ]
        )

        pd.testing.assert_frame_equal(output_df, expected_output)

class TestFittingFunction:
    '''
    Test fits
    '''
    def test_prepare_fit_data(self):

        input_df = pd.DataFrame(
            columns = ['p1','p2','t1','t2', 'avgs', 'N', 'dt', 'freqs', 'percentiles'],
            data = [
                ['75', '50', 'time1', 'time3', 0, 30, 3, 4.5, '75-50'],
                ['75', '50', 'time1', 'time3', 1, 30, 3, 4.5, '75-50'],
                ['100', '75', 'time1', 'time3', 5, 50, 2, 5.5, '100-75'],
                ['75', '50', 'time1', 'time3', 2, 30, 3, 4.5, '75-50'],
                ['75', '50', 'time1', 'time3', 0, 30, 3, 4.5, '75-50'],
            ]
        )

        expected_f = np.array([4.5,5.5,4.5])
        expected_avg = np.array([1, 5, 2])

        fit_f, fit_a = params.prepare_fit_data(input_df)

        np.testing.assert_array_equal(expected_f, fit_f)
        np.testing.assert_array_equal(expected_avg, fit_a)

    def test_fitting(self):
        x = np.linspace(1,100, 50)
        b = 1.5
        s = .8

        y = b*x + (s*x)**2

        fit_s, fit_b = params._get_fit(x, y)
        np.testing.assert_almost_equal(fit_s, s)
        np.testing.assert_almost_equal(fit_b, b)

    def test_fitting2(self):
        x = np.logspace(1e-6,1e-1, 500)
        b = 1.5/1e4
        s = .8

        y = b*x + (s*x)**2

        fit_s, fit_b = params._get_fit(x, y)
        np.testing.assert_almost_equal(fit_s, s)
        np.testing.assert_almost_equal(fit_b, b)

    def test_fitting3(self):
        x = np.logspace(1e-6,1e-1, 500)
        b = 1.5/27562
        s = 0.3 # this fails if >1 because of the set bounds

        y = b*x + (s*x)**2

        fit_s, fit_b = params._get_fit(x, y)
        np.testing.assert_almost_equal(fit_s, s)
        np.testing.assert_almost_equal(fit_b, b)
