'''
Test functions to calculate noise
'''

import pandas as pd
import numpy as np
import pytest
from clonesearch.utils import noise_functions as noise

class TestPercentileCalculation:
    def test_get_vals_for_perc(self):
        df_freq = pd.DataFrame(
            data = [
                [0,5,4],
                [5,5,5],
                [1,3,7],
                [1,0,0]
                ],
            columns = ['time1','time2','time3'],
            index = ['c1','c2','c3','c4']
        )

        expected = np.array([[0,5],[5,5],[1,3],[1,0]])
        vals, mask, means = noise._get_vals_for_perc(df_freq,['time1','time2'])
        np.testing.assert_array_equal(vals, expected)
        means_exp = np.array([2.5,5,2,.5])
        np.testing.assert_array_equal(means, means_exp)
        mask_exp = [True,True,True,True]
        np.testing.assert_array_equal(mask, np.array(mask_exp))

        expected = np.array([[0,4],[5,5],[1,7],[1,0]])
        vals, mask, means = noise._get_vals_for_perc(df_freq,['time1','time3'])
        np.testing.assert_array_equal(vals, expected)
        means_exp = np.array([2,5,4,.5])
        np.testing.assert_array_equal(means, means_exp)
        mask_exp = [True,True,True,True]
        np.testing.assert_array_equal(mask, np.array(mask_exp))

        expected = np.array([[5,4],[5,5],[3,7]])
        vals, mask, means = noise._get_vals_for_perc(df_freq,['time2','time3'])
        np.testing.assert_array_equal(vals, expected)
        means_exp = np.array([4.5,5,5])
        np.testing.assert_array_equal(means, means_exp)
        mask_exp = [True,True,True,False]
        np.testing.assert_array_equal(mask, np.array(mask_exp))

    def test_get_percentile_values(self):

        means = np.array([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
        perc_step = 20

        x,y = noise._get_percentile_values(means, perc_step)
        assert x == [('20','0'),('40','20'),('60','40'),('80','60'),('100','80')]
        assert y == [(.2,0),(.4,.2),(.6,.4),(.8,.6),(1,.8)]

    def test_get_sets(self):

        df_freq = pd.DataFrame(
            data = [
                [0,5,4],
                [5,5,5],
                [1,3,7],
                [1,5,0]
                ],
            columns = ['time1','time2','time3'],
            index = ['c1','c2','c3','c4']
        )

        mask = np.array([False,True,True,True])
        percentile_vals = [(3, 0), (5, 3)]
        percentiles = [('.5','0'), ('1','.5')]

        sets = noise._get_sets(df_freq, ['time1', 'time2'], percentiles, percentile_vals, mask)

        expected_set1 = np.array([False, False, True, True])
        expected_set2 = np.array([False, True, False, False])
        np.testing.assert_array_equal(sets[0], expected_set1)
        np.testing.assert_array_equal(sets[1], expected_set2)

class TestDifferenceCalculation:

    @pytest.fixture(autouse=True)
    def setUp(self):
        self.df_freq = pd.DataFrame(
                        data = [
                            [0,5,4],
                            [5,5,5],
                            [1,3,7],
                            [1,0,0]
                            ],
                        columns = ['time1','time2','time3'],
                        index = ['c1','c2','c3','c4']
                    )

        self.df_freq1 = pd.DataFrame(
                        data = [
                            [0,5,4],
                            [5,5,5],
                            [1,3,7],
                            [2,3,7],
                            [5,4,7],
                            [1,0,0]
                            ],
                        columns = ['time1','time2','time3'],
                        index = ['c1','c2','c3','c4','c5','c6']
                    )

        self.tp_dict = {
                'time1': 7,
                'time2': 15,
                'time3': 85
            }

        self.tp_names = ['time1','time2','time3']
        self.tp_pairs = [('time1', 'time2'), ('time1', 'time3'), ('time2', 'time3')]

    def test_no_percentile_output(self):

        dt_min = 0
        outputs, dt, freqs = noise._no_percentile_output(
                                        noise.var_btn_tps, self.df_freq,
                                        self.tp_pairs, self.tp_dict, dt_min
                                        )

        expected_output = [
            (25+0+4+1)/4,
            (16+0+36+1)/4,
            (1+0+16)/3
        ]
        np.testing.assert_almost_equal(outputs, expected_output)

        expected_dt = [8,78,70]
        np.testing.assert_almost_equal(dt, expected_dt)

        expected_freqs = [2.5,23/8,29/6] # do not consider value that is 0 in both tps
        np.testing.assert_almost_equal(freqs, expected_freqs)

    def test_no_percentile_output_w_dt_min(self):

        dt_min = 10
        outputs, dt, freqs = noise._no_percentile_output(
                                noise.var_btn_tps, self.df_freq,
                                self.tp_pairs, self.tp_dict, dt_min
                                )

        expected_output = [
            (16+0+36+1)/4,
            (1+0+16)/3
        ]
        np.testing.assert_almost_equal(outputs, expected_output)

        expected_dt = [78,70]
        np.testing.assert_almost_equal(dt, expected_dt)

        expected_freqs = [23/8,29/6] # do not consider value that is 0 in both tps
        np.testing.assert_almost_equal(freqs, expected_freqs)

    def test_get_single_perc_output(self):

        ss = [False,True,False,False]
        t1 = 'time1'
        t2 = 'time2'

        o, f = noise._get_single_perc_output(noise.var_btn_tps, self.df_freq, t1, t2, ss)

        expected_output = 0
        expected_freqs = 5
        assert o == expected_output
        assert f == expected_freqs

        ss = [True,True,True,True]
        o, f = noise._get_single_perc_output(noise.var_btn_tps, self.df_freq, t1, t2, ss)

        expected_output = (25+0+4+1)/4
        expected_freqs = 20/8
        assert o == expected_output
        assert f == expected_freqs

    def test_get_percentiles_output(self):

        sets = [
            [False,True,False,False],
            [True,True,True,True],
            [False,False,False,False],
            [True,False,True,True]
        ]
        t1 = 'time1'
        t2 = 'time2'

        percentiles = [('.25','0'), ('.5','.25'), ('.75','.5'),('1','.75')]
        min_perc = 0

        dt_dict, N_dict, o_dict, f_dict = noise._get_percentiles_output(
                                                noise.var_btn_tps, sets, percentiles,
                                                min_perc, t1, t2, self.tp_dict, self.df_freq
                                                )

        expected_dt_dict = {
            ('.25','0'):8,
            ('.5','.25'):8,
            # note this percentile not calculated because all empty
            ('1','.75'):8
        }
        assert expected_dt_dict == dt_dict

        expected_N_dict = {
            ('.25','0'):1,
            ('.5','.25'):4,
            # note this percentile not calculated because all empty
            ('1','.75'):3
        }
        assert expected_N_dict == N_dict

        expected_o_dict = {
            ('.25','0'):0,
            ('.5','.25'):(25+0+4+1)/4,
            # note this percentile not calculated because all empty
            ('1','.75'):(25+4+1)/3
        }
        assert expected_o_dict == o_dict

        expected_f_dict = {
            ('.25','0'):5,
            ('.5','.25'):20/8,
            # note this percentile not calculated because all empty
            ('1','.75'):10/6
        }
        assert expected_f_dict == f_dict


    def test_get_percentiles_output_w_min_perc(self):
        sets = [
            [False,True,False,False],
            [True,True,True,True],
            [False,False,False,False],
            [True,False,True,True]
        ]
        t1 = 'time1'
        t2 = 'time2'

        percentiles = [('.25','0'), ('.5','.25'), ('.75','.5'),('1','.75')]
        min_perc = 0.6

        dt_dict, N_dict, o_dict, f_dict = noise._get_percentiles_output(
                                                noise.var_btn_tps, sets, percentiles,
                                                min_perc, t1, t2, self.tp_dict, self.df_freq
                                                )

        expected_dt_dict = {
            ('1','.75'):8
        }
        assert expected_dt_dict == dt_dict

        expected_N_dict = {
            ('1','.75'):3
        }
        assert expected_N_dict == N_dict

        expected_o_dict = {
            ('1','.75'):(25+4+1)/3
        }
        assert expected_o_dict == o_dict

        expected_f_dict = {
            ('1','.75'):10/6
        }
        assert expected_f_dict == f_dict

    def test_complete_calc_diffs_function_no_perc(self):

        outputs, dt, freqs = noise.calc_diffs(
                                noise.var_btn_tps, self.df_freq, self.tp_names,
                                self.tp_dict, use_percentiles = False, dt_min = 0
                                )

        expected_output = [
            (25+0+4+1)/4,
            (16+0+36+1)/4,
            (1+0+16)/3
        ]
        np.testing.assert_almost_equal(outputs, expected_output)

        expected_dt = [8,78,70]
        np.testing.assert_almost_equal(dt, expected_dt)

        expected_freqs = [2.5,23/8,29/6]
        np.testing.assert_almost_equal(freqs, expected_freqs)

    def test_complete_calc_diffs_function_w_perc(self):

        outputs, dt, freqs, percentiles, N = noise.calc_diffs(
                                                noise.var_btn_tps,
                                                self.df_freq1, self.tp_names, self.tp_dict,
                                                use_percentiles = True, perc_step =25, dt_min = 0
                                                )

        assert percentiles == [('25','0'), ('50','25'), ('75','50'), ('100','75')]

        expected_N = {
            ('time1','time2'): {
                                ('25','0') :  2,
                                ('50','25'):  2,
                                ('75','50'):  1,
                                ('100','75'): 1
                            },
            ('time1','time3'): {
                                ('25','0') :  2,
                                ('50','25'):  1,
                                ('75','50'):  1,
                                ('100','75'): 2
                            },
            ('time2','time3'): {
                                ('25','0') :  1,
                                ('50','25'):  3,
                                ('100','75'): 1
                            },
        }
        assert N == expected_N

        expected_dt = {
            ('time1','time2'): {
                                ('25','0') :  8,
                                ('50','25'):  8,
                                ('75','50'):  8,
                                ('100','75'): 8
                            },
            ('time1','time3'): {
                                ('25','0') :  78,
                                ('50','25'):  78,
                                ('75','50'):  78,
                                ('100','75'): 78
                            },
            ('time2','time3'): {
                                ('25','0') :  70,
                                ('50','25'):  70,
                                ('100','75'): 70
                            },
        }
        assert dt == expected_dt

        expected_output = {
            ('time1','time2'): {
                                ('25','0') :  (2**2 + 1**2)/2,
                                ('50','25'):  (5**2 + 1**2)/2,
                                ('75','50'):  (1**2),
                                ('100','75'): (0**2)
                            },
            ('time1','time3'): {
                                ('25','0') :  (4**2 + 1**2)/2,
                                ('50','25'):  36,
                                ('75','50'):  25,
                                ('100','75'): (4 + 0)/2
                            },
            ('time2','time3'): {
                                ('25','0') :  (1**2),
                                ('50','25'):  (0**2 + 4**2 + 4**2)/3,
                                ('100','75'): 3**2
                            },
        }
        assert outputs == expected_output

        expected_freqs = {
            ('time1','time2'): {
                                ('25','0') :  (0.5 + 2)/2,
                                ('50','25'):  (2.5+2.5)/2,
                                ('75','50'):  4.5,
                                ('100','75'): 5
                            },
            ('time1','time3'): {
                                ('25','0') :  (.5+2)/2,
                                ('50','25'):  4.,
                                ('75','50'):  4.5,
                                ('100','75'): (5+6)/2
                            },
            ('time2','time3'): {
                                ('25','0') :  4.5,
                                ('50','25'): (5+5+5)/3,
                                ('100','75'): 5.5
                            },
        }
        assert freqs == expected_freqs

    def test_complete_calc_diffs_function_w_perc_min_perc(self):

        outputs, dt, freqs, percentiles, N = noise.calc_diffs(
                                                noise.var_btn_tps,
                                                self.df_freq1, self.tp_names, self.tp_dict,
                                                use_percentiles = True, perc_step =25,
                                                min_perc = 50, dt_min = 0
                                                )

        assert percentiles == [('75','50'), ('100','75')]

        expected_N = {
            ('time1','time2'): {
                                ('75','50'):  1,
                                ('100','75'): 1
                            },
            ('time1','time3'): {
                                ('75','50'):  1,
                                ('100','75'): 2
                            },
            ('time2','time3'): {
                                ('100','75'): 1
                            },
        }
        assert N == expected_N

        expected_dt = {
            ('time1','time2'): {
                                ('75','50'):  8,
                                ('100','75'): 8
                            },
            ('time1','time3'): {
                                ('75','50'):  78,
                                ('100','75'): 78
                            },
            ('time2','time3'): {
                                ('100','75'): 70
                            },
        }
        assert dt == expected_dt

        expected_output = {
            ('time1','time2'): {
                                ('75','50'):  ((1)**2),
                                ('100','75'): ((0)**2)
                            },
            ('time1','time3'): {
                                ('75','50'):  25,
                                ('100','75'): (4 + 0)/2
                            },
            ('time2','time3'): {
                                ('100','75'): (3)**2
                            },
        }
        assert outputs == expected_output

        expected_freqs = {
            ('time1','time2'): {
                                ('75','50'):  4.5,
                                ('100','75'): 5
                            },
            ('time1','time3'): {
                                ('75','50'):  4.5,
                                ('100','75'): (5+6)/2
                            },
            ('time2','time3'): {
                                ('100','75'): 5.5
                            },
        }
        assert freqs == expected_freqs

    def test_complete_calc_diffs_function_w_perc_min_perc_min_dt(self):

        outputs, dt, freqs, percentiles, N = noise.calc_diffs(
                                                noise.var_btn_tps,
                                                self.df_freq1, self.tp_names, self.tp_dict,
                                                use_percentiles = True, perc_step =25,
                                                min_perc = 50, dt_min = 75
                                                )

        assert percentiles == [('75','50'), ('100','75')]

        expected_N = {
            ('time1','time3'): {
                                ('75','50'):  1,
                                ('100','75'): 2
                            },
        }
        assert N == expected_N

        expected_dt = {
            ('time1','time3'): {
                                ('75','50'):  78,
                                ('100','75'): 78
                            },
        }
        assert dt == expected_dt

        expected_output = {
            ('time1','time3'): {
                                ('75','50'):  25,
                                ('100','75'): (4 + 0)/2
                            },
        }
        assert outputs == expected_output

        expected_freqs = {
            ('time1','time3'): {
                                ('75','50'):  4.5,
                                ('100','75'): (5+6)/2
                            },
        }
        assert freqs == expected_freqs

    def test_complete_calc_diffs_function_w_perc_diff_perc(self):

        outputs, dt, freqs, percentiles, N = noise.calc_diffs(
                                                noise.var_btn_tps,
                                                self.df_freq1, self.tp_names, self.tp_dict,
                                                use_percentiles = True, perc_step =20,
                                                min_perc = 50, dt_min = 75
                                                )

        assert percentiles == [('70','50'), ('90','70'), ('100','90')]

class TestMakeDf:
    def test_make_df_simple(self):

        input_freqs = {
            ('time1','time3'): {
                                ('75','50'):  4.5,
                                ('100','75'): 5.5
                            },
        }

        output_df = noise.make_df(input_freqs, 'freqs')

        expected_output = pd.DataFrame(
            columns = ['p1','p2','t1','t2', 'freqs'],
            data = [
                ['75', '50', 'time1', 'time3', 4.5],
                ['100', '75', 'time1', 'time3', 5.5]
            ]
        )

        pd.testing.assert_frame_equal(
            output_df.sort_values(['p1', 'p2', 't1','t2']).reset_index(drop=True),
            expected_output.sort_values(['p1', 'p2', 't1','t2']).reset_index(drop=True)
        )

    def test_make_df_more_complex(self):

        input_freqs = {
            ('time1','time2'): {
                                ('75','50'):  4.5,
                                ('100','75'): 5
                            },
            ('time1','time3'): {
                                ('75','50'):  4.5,
                                ('100','75'): 5.5
                            },
            ('time2','time3'): {
                                ('100','75'): 5.5
                            },
        }

        output_df = noise.make_df(input_freqs, 'something_silly').reset_index(drop = True)

        expected_output = pd.DataFrame(
            columns = ['p1','p2','t1','t2', 'something_silly'],
            data = [
                ['75', '50', 'time1', 'time2', 4.5],
                ['100', '75', 'time1', 'time2', 5],
                ['75', '50', 'time1', 'time3', 4.5],
                ['100', '75', 'time1', 'time3', 5.5],
                ['100', '75', 'time2', 'time3', 5.5]
            ]
        )

        pd.testing.assert_frame_equal(
            output_df.sort_values(['p1', 'p2', 't1','t2']).reset_index(drop=True),
            expected_output.sort_values(['p1', 'p2', 't1','t2']).reset_index(drop=True)
        )

    def test_make_df_withNA(self):

        input_freqs = {
            ('time1','time2'): {
                                ('75','50'):  4.5,
                                ('100','75'): np.nan
                            },
            ('time1','time3'): {
                                ('75','50'):  4.5,
                                ('100','75'): 5.5
                            },
            ('time2','time3'): {
                                ('100','75'): 5.5
                            },
        }

        output_df = noise.make_df(input_freqs, 'something_silly').reset_index(drop = True)

        expected_output = pd.DataFrame(
            columns = ['p1','p2','t1','t2', 'something_silly'],
            data = [
                ['75', '50', 'time1', 'time2', 4.5],
                ['75', '50', 'time1', 'time3', 4.5],
                ['100', '75', 'time1', 'time3', 5.5],
                ['100', '75', 'time2', 'time3', 5.5]
            ]
        )

        pd.testing.assert_frame_equal(
            output_df.sort_values(['p1', 'p2', 't1','t2']).reset_index(drop=True),
            expected_output.sort_values(['p1', 'p2', 't1','t2']).reset_index(drop=True)
        )
