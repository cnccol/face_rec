import os
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

def key_params():
    
    door = 1
    n_jobs = -1
    eps = 0.39
    min_samples = 5
    max_h_face = 182
    min_rgb_mean = 20
    min_hog_score = -0.4
    face_types = [0, 1, 2]
    h_face_inside = 120
    worker_visits_threshold = 2

    return (door, n_jobs, eps, min_samples, max_h_face, min_rgb_mean, 
            min_hog_score, face_types, h_face_inside, worker_visits_threshold)

def append_and_save(door):

    CSVs_dir = 'CSVs_cnn_hog'
    encodings_dir = 'encodings_cnn_hog'

    dates = []
    for csv in sorted(os.listdir(CSVs_dir)):
        date = csv[:-6]
        if date >= "2019-10-31":
            dates.append(date)

    name = dates[0] + '_to_' + dates[-1]
    print(name)

    dfs_days = []
    encodings_days = []

    for date in dates:
            
        b = pd.read_csv(os.path.join(CSVs_dir, (date + '_1.csv')), 
                        index_col=0)
        b['Concesionario'] = door
        e = np.load(os.path.join(encodings_dir, (date + '_1.npy')))
        
        dfs_days.append(b)
        encodings_days.append(e)
            
    df_faces = pd.concat(dfs_days, ignore_index=True)
    encodings = np.concatenate(encodings_days, axis=0)

    df_faces['mean_RGB'] = (df_faces['RGB'].str[1: -1].str.split(expand=True)
                           .astype(np.float64).mean(axis=1))

    # df_faces.to_csv(os.path.join('for_clustering', 
    #                          (str(door) + '_' + name + '.csv')))
    # np.save(os.path.join('for_clustering', (str(door) + '_' + name + '.npy')), 
    #         encodings)
    
    return df_faces, encodings, name

def cluster_faces(df_faces, encodings, n_jobs, eps, min_samples, max_h_face, 
                  min_rgb_mean, min_hog_score, face_types, h_face_inside):
    tt = time.time()
    filter = ((df_faces.h_face <= max_h_face) &
              (df_faces.mean_RGB >= min_rgb_mean) &
              (df_faces.score >= min_hog_score) & 
              (df_faces.face_type.isin(face_types)))
    filtered_df = df_faces[filter].reset_index(drop=True)

    filtered_df['jump_date'] = False
    filtered_df.loc[1:, 'jump_date'] = [True if (filtered_df.loc[i, 'date'] 
                                        > filtered_df.loc[i-1, 'date']) else 
                                        False for i in filtered_df.index[1:]]

    filtered_encodings = encodings[filter]

    cluster_endings = list(filtered_df[filtered_df.jump_date].index
                          ) + [filtered_df.index[-1]+1]
 
    filtered_df['match'] = 0
    
    start = 0
    end = cluster_endings[0]

    clt = DBSCAN(eps=eps, min_samples=min_samples,n_jobs=n_jobs)

    clt.fit(filtered_encodings[start:end, :])
    
    matched_df = filtered_df.loc[start:end-1, :].reset_index(drop=True) 
    matched_df = matched_df.loc[clt.core_sample_indices_, 
                                :].reset_index(drop=True) 
    matched_df['match'] = clt.labels_[clt.core_sample_indices_]
    
    visitors = set([matched_df.loc[i, 'match'] for i in matched_df.index if
                    matched_df.loc[i, 'h_face'] > h_face_inside])

    visitors_boolean = matched_df['match'].isin(visitors)
    matched_df = matched_df[visitors_boolean].reset_index(drop=True)
    
    core_encodings = clt.components_[visitors_boolean]
    
    print(end, len(matched_df), core_encodings.shape, 
          matched_df.loc[len(matched_df)-1, 'date'], time.time() - tt)
    tt = time.time()
    start = end

    for end in cluster_endings[1:]:
        clt = DBSCAN(eps=eps, min_samples=min_samples,n_jobs=n_jobs)

        clt.fit(np.append(core_encodings, filtered_encodings[start:end, :], 
                          axis=0).reshape(-1, 128))
        
        matched_df = matched_df.append(filtered_df.loc[start:end-1, :], 
                                       ignore_index=True)
        matched_df = matched_df.loc[clt.core_sample_indices_, 
                                    :].reset_index(drop=True)
        matched_df['match'] = clt.labels_[clt.core_sample_indices_]

        visitors = set([matched_df.loc[i, 'match'] for i in matched_df.index if
                        matched_df.loc[i, 'h_face'] > h_face_inside])

        visitors_boolean = matched_df['match'].isin(visitors)
        matched_df = matched_df[visitors_boolean].reset_index(drop=True)

        core_encodings = clt.components_[visitors_boolean]

        print(end, len(matched_df), core_encodings.shape, 
              matched_df.loc[len(matched_df)-1, 'date'], time.time() - tt)
        tt = time.time()
        start = end
        
    return matched_df, core_encodings

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
    # workers = visits_df.person.value_counts()[visits_df.person.value_counts() 
    #                                           > worker_visits_threshold].index
    visitors_df = visits_df[visits_df.person.isin(visitors)]
    # workers_df = visits_df[visits_df.person.isin(workers)]

    visitors_df = visitors_df[['date', 'time', 'person', 'Concesionario', 
                               'gender']]
    visitors_df.columns = ['fecha', 'hora', 'persona', 'Concesionario', 'Sexo']

    return visitors_df
            

def main():

    (door, n_jobs, eps, min_samples, max_h_face, min_rgb_mean,
     min_hog_score, face_types, h_face_inside, 
     worker_visits_threshold) = key_params()

    df_faces, encodings, name = append_and_save(door)

    matched_df, core_visits_encodings = cluster_faces(df_faces, encodings, 
                                                      n_jobs, eps, min_samples, 
                                                      max_h_face, min_rgb_mean, 
                                                      min_hog_score, 
                                                      face_types, h_face_inside
                                                     )

    matched_df.to_csv('core_CSVs/' + name + '.csv')
    np.save('core_encodings/' + name + '.npy', core_visits_encodings)

    visits_df = make_visits_df(matched_df, worker_visits_threshold)

    visits_df.to_csv('tabla_visitas_' + str(door) + '.csv')

if __name__ == '__main__':
    main()
