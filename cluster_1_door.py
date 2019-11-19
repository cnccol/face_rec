import os
import configparser
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

def main(up_to_date):
    (door, n_jobs, eps, min_samples, max_h_face, min_rgb_mean,
     min_hog_score, face_types, h_face_inside,
     worker_visits_threshold) = parse_cfg()

    core_df = pd.read_csv('core_CSVs/' + sorted(os.listdir('core_CSVs'))[-1],
                           index_col=0)
    core_npy = np.load(
        'core_encodings/' + sorted(os.listdir('core_encodings'))[-1])

    CSVs_dir = 'CSVs_cnn_hog'
    encodings_dir = 'encodings_cnn_hog'

    dates = []
    for csv in sorted(os.listdir(CSVs_dir)):
        date = csv[:-6]
        if core_df['date'].max() < date <= up_to_date:
            dates.append(date)

    clustering_dates = dates[0] + '_to_' + dates[-1]
    print('Clustering:',  clustering_dates)

    for date in dates:
        tt = time.time()

        df_day = pd.read_csv(os.path.join(CSVs_dir, (date + '_1.csv')),
                        index_col=0)
        df_day['Concesionario'] = door
        df_day['mean_RGB'] = (df_day['RGB'].str[1: -1].str.split(expand=True)
                              .astype(np.float64).mean(axis=1))

        npy_day = np.load(os.path.join(encodings_dir, (date + '_1.npy')))

        filter_faces = ((df_day.h_face <= max_h_face) &
                        (df_day.mean_RGB >= min_rgb_mean) &
                        (df_day.score >= min_hog_score) &
                        (df_day.face_type.isin(face_types)))
        df_day = df_day[filter_faces].reset_index(drop=True)
        npy_day = npy_day[filter_faces]

        filter_11_days = (core_df.date >= sorted(core_df.date.unique())[-11])
        core_df = core_df[filter_11_days].reset_index(drop=True)
        core_npy = core_npy[filter_11_days]

        core_df = core_df[df_day.columns]
        core_df = pd.concat([core_df, df_day], ignore_index=True)
        core_npy = np.concatenate([core_npy, npy_day], axis=0)

        clt = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs)

        clt.fit(core_npy)

        core_indices = clt.core_sample_indices_
        core_df = core_df.loc[core_indices, :].reset_index(drop=True)
        core_df['match'] = clt.labels_[core_indices]

        visitors = set([core_df.loc[i, 'match'] for i in core_df.index if
                        core_df.loc[i, 'h_face'] > h_face_inside])

        visitors_boolean = core_df['match'].isin(visitors)
        core_df = core_df[visitors_boolean].reset_index(drop=True)

        core_npy = clt.components_[visitors_boolean]

        print(core_df.loc[len(core_df)-1, 'date'], time.time() - tt)

    name = core_df.date.min() + '_to_' core_df.date.max()

    core_df.to_csv('core_CSVs/' + name + '.csv')
    np.save('core_encodings/' + name + '.npy', core_npy)

    print(name, 'done and saved')

    weekday = datetime.strptime(up_to_date, '%Y-%m-%d').weekday()

    if weekday == 6:
        visits_df = make_visits_df(core_df, worker_visits_threshold)

        visits_df.to_csv('tablas_visitas/' + name + '.csv')

        print('Also, visits_df done and saved')


def parse_cfg():
    config = configparser.ConfigParser()
    config.read('cfg.ini')

    door = int(config['cluster']['door'])
    n_jobs = int(config['cluster']['n_jobs'])
    eps = float(config['cluster']['eps'])
    min_samples = int(config['cluster']['min_samples'])
    max_h_face = int(config['cluster']['max_h_face'])
    min_rgb_mean = int(config['cluster']['min_rgb_mean'])
    min_hog_score = int(config['cluster']['min_hog_score'])
    face_types = eval(config['cluster']['face_types'])
    h_face_inside = int(config['cluster']['h_face_inside'])
    worker_visits_threshold = int(config['cluster']['worker_visits_threshold'])

    return (door, n_jobs, eps, min_samples, max_h_face, min_rgb_mean,
            min_hog_score, face_types, h_face_inside, worker_visits_threshold)


def make_visits_df(matched_df, worker_visits_threshold):
    sex_dict = {}
    for person in np.sort(matched_df.match.unique()):
        argmax = matched_df.loc[matched_df['match'] == person,
                                'gender_conf'].idxmax()
        sex_dict[person] = matched_df.loc[argmax, 'gender']

    dict_last_visits = {person: '2019-05-01' for person in
                        np.sort(matched_df.match.unique())}
    visits_df = pd.DataFrame(columns = ['date', 'frame', 'person'])

    for i in matched_df.index:
        match = matched_df.loc[i, 'match']
        if match == -1:
            continue
        date = matched_df.loc[i, 'date']
        concesionario = matched_df.loc[i, 'Concesionario']
        if dict_last_visits[match] < date:
            visits_df = visits_df.append({'date': date,
                                          'frame': matched_df.loc[i, 'frame'],
                                          'person': match,
                                          'gender': sex_dict[match],
                                          'Concesionario': concesionario},
                                         ignore_index=True)
        dict_last_visits[match] = date
    for i in visits_df.index:
        frame = visits_df.loc[i, 'frame']
        visits_df.loc[i, 'time'] = str((datetime(2000, 1, 1, 9, 0, 0, 0)
                                        + timedelta(seconds=(frame*0.08))
                                       ).time())

    visitors = visits_df.person.value_counts()[visits_df.person.value_counts()
                                               <= worker_visits_threshold].index

    visitors_df = visits_df[visits_df.person.isin(visitors)]

    visitors_df = visitors_df[['date', 'time', 'person', 'Concesionario',
                               'gender']]
    visitors_df.columns = ['fecha', 'hora', 'persona', 'Concesionario', 'Sexo']

    return visitors_df


if __name__ == '__main__':
    main('2019-11-11')
