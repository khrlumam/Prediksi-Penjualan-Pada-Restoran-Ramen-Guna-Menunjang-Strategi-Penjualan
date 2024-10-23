import os
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pickle
import math
from function import dict_diskon, dict_menu, dict_week, month_weeks_dict
import mpld3
from IPython.display import HTML

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

#upload files
app.config['UPLOAD_FOLDER']='uploads'
ALLOWED_EXTENSIONS = {'xlsx'}

def allowed_file(filename):
  return '.' in filename and \
    filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload-file', methods=['GET', 'POST'])
def upload_file():
  if request.method == 'GET':
    return render_template('index.html')
  
  elif request.method == 'POST':
    # check if the post request has the file part
    if 'file' not in request.files:
      return redirect(request.url)
    
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
      return redirect(request.url)
    
  if file and allowed_file(file.filename):
    file = request.files['file']
    file.filename = "dataset.xlsx"
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    return render_template('index.html')
      
      

@app.route('/prediksi-rekomendasi', methods=['GET', 'POST'])
def page():
  return render_template ('prediksidanrekomendasi.html')


@app.route('/prediksi-rekomendasi/result', methods=['GET','POST'])
def predrek():
  df = pd.read_excel('uploads/dataset.xlsx')
  data = df[['nama_item', 'week', 'jenis_diskon','item_terjual']]
  Q1 = np.percentile(data['item_terjual'], 25,
                   interpolation = 'midpoint')

  Q3 = np.percentile(data['item_terjual'], 75,
                    interpolation = 'midpoint')
  IQR = Q3 - Q1

  # Upper bound
  upper = np.where(data['item_terjual'] >= (Q3+1.5*IQR))
  # Lower bound
  lower = np.where(data['item_terjual']<= (Q1-1.5*IQR))

  ''' Removing the Outliers '''
  data.drop(upper[0], inplace = True)
  data.drop(lower[0], inplace = True)
  
  label_encoders = {}
  cat_cols = ['nama_item', 'week', 'jenis_diskon']
  for col in cat_cols:
      label_encoders[col] = LabelEncoder()
      label_encoders[col].fit(data[col])
      data[col] = label_encoders[col].transform(data[col])

  X = data.drop('item_terjual', axis=1)
  y = data['item_terjual']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Step 4: Model Training
  xgbr = xgb.XGBRegressor()
  xgbr.fit(X_train, y_train)

  # Create an empty list to store the loss values
  losses = []

  # Create the XGBoost model
  xgbr = xgb.XGBRegressor()

  # Train the model
  xgbr.fit(X_train, y_train)

  # Calculate the initial loss
  y_pred = xgbr.predict(X_train)
  initial_loss = np.sqrt(mean_squared_error(y_train, y_pred))
  losses.append(initial_loss)

  # Perform additional training iterations
  num_iterations = 10
  for i in range(num_iterations):
      xgbr.fit(X_train, y_train, xgb_model=xgbr)
      y_pred = xgbr.predict(X_train)
      loss = np.sqrt(mean_squared_error(y_train, y_pred))
      losses.append(loss)

  pickle.dump(xgbr, open('uploads/xgbr.model','wb'))
  week = int(request.form['kind'])
  df_temp = data.loc[data['week']==week]
  df_avg = df_temp['item_terjual'].mean()
  df_temp = df_temp.loc[df_temp['item_terjual']<df_avg]
  nama_menu= df_temp.nama_item.unique()
  diskon = data.jenis_diskon.unique()
  menu = dict_menu()
  jenisdiskon = dict_diskon()
  minggu = dict_week()
  periode = minggu[week]
  df_out = pd.DataFrame(columns=['Nama Makanan', 'Rekomendasi Jenis Diskon', 'Jumlah Prediksi Terjual'])

  for i in nama_menu:
    x = [] 
    output_messages = []
    for j in diskon:
      xtest = [[i, week, j]]
      df_test = pd.DataFrame(xtest, columns=['nama_item', 'week', 'jenis_diskon'])
      pred = xgbr.predict(df_test)
      x.append(pred[0])  
      nilai_terbesar = x[0]
      for n in x:
        if n > nilai_terbesar:
          nilai_terbesar = n
    output_messages.append((menu[i], jenisdiskon[diskon[x.index(nilai_terbesar)]],math.floor(nilai_terbesar)))
    df_temp = pd.DataFrame(output_messages, columns=['Nama Makanan', 'Rekomendasi Jenis Diskon', 'Jumlah Prediksi Terjual'])
    df_out = pd.concat([df_out, df_temp], ignore_index=True)
  rata = df_out['Jumlah Prediksi Terjual'].mean()
  z=[]

  for k in df_out['Jumlah Prediksi Terjual']:
      if k > rata:
          num_stars = 3
      elif k == rata:
          num_stars = 2
      else:
          num_stars = 1
      
      if num_stars > 0:
          html_code = ''
          for _ in range(num_stars):
              html_code += '★'  # Unicode character for a star
          z.append(html_code)
  df_out['Rating Prediksi'] = z
  data=df_out.to_html()
  print(data)

  return render_template('prediksidanrekomendasi.html', periode=periode, data=df_out.to_html(justify='center'))
    
@app.route('/prediksi', methods=['GET', 'POST'])
def pred():
  return render_template ('prediksi.html')

@app.route('/prediksi/result', methods=['GET', 'POST'])
def tesmodel():
  df = pd.read_excel('uploads/dataset.xlsx')
  data = df[['nama_item', 'week', 'jenis_diskon','item_terjual']]
  Q1 = np.percentile(data['item_terjual'], 25,
                   interpolation = 'midpoint')

  Q3 = np.percentile(data['item_terjual'], 75,
                    interpolation = 'midpoint')
  IQR = Q3 - Q1

  # Upper bound
  upper = np.where(data['item_terjual'] >= (Q3+1.5*IQR))
  # Lower bound
  lower = np.where(data['item_terjual']<= (Q1-1.5*IQR))

  ''' Removing the Outliers '''
  data.drop(upper[0], inplace = True)
  data.drop(lower[0], inplace = True)
  
  label_encoders = {}
  cat_cols = ['nama_item', 'week', 'jenis_diskon']
  for col in cat_cols:
      label_encoders[col] = LabelEncoder()
      label_encoders[col].fit(data[col])
      data[col] = label_encoders[col].transform(data[col])

  X = data.drop('item_terjual', axis=1)
  y = data['item_terjual']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Step 4: Model Training
  xgbr = xgb.XGBRegressor()
  xgbr.fit(X_train, y_train)

  # Create an empty list to store the loss values
  losses = []

  # Create the XGBoost model
  xgbr = xgb.XGBRegressor()

  # Train the model
  xgbr.fit(X_train, y_train)

  # Calculate the initial loss
  y_pred = xgbr.predict(X_train)
  initial_loss = np.sqrt(mean_squared_error(y_train, y_pred))
  losses.append(initial_loss)

  # Perform additional training iterations
  num_iterations = 10
  for i in range(num_iterations):
      xgbr.fit(X_train, y_train, xgb_model=xgbr)
      y_pred = xgbr.predict(X_train)
      loss = np.sqrt(mean_squared_error(y_train, y_pred))
      losses.append(loss)

  pickle.dump(xgbr, open('uploads/xgbr.model','wb'))
  week = int(request.form['kind'])
  nama_menu = int(request.form['menu'])
  diskon = data.jenis_diskon.unique()
  menu = dict_menu()
  jenisdiskon = dict_diskon()
  output_messages = []
  minggu = dict_week()
  periode = minggu[week]
  df_out = pd.DataFrame(columns=['Nama Makanan', 'Rekomendasi Jenis Diskon', 'Jumlah Prediksi Terjual'])

  for n in diskon :
    xtest = [[nama_menu,week,n]]
    df_test = pd.DataFrame(xtest, columns=[['nama_item','week','jenis_diskon']])
    pred = xgbr.predict(df_test)
    if n==4 :
      output_messages.append((menu[nama_menu], jenisdiskon[n],math.floor(pred)))
    elif n==0 :
      output_messages.append((menu[nama_menu], jenisdiskon[n],math.floor(pred)))
    elif n==1 :
      output_messages.append((menu[nama_menu], jenisdiskon[n],math.floor(pred)))
    elif n==2 :
      output_messages.append((menu[nama_menu], jenisdiskon[n],math.floor(pred)))
    elif n==3 :
      output_messages.append((menu[nama_menu], jenisdiskon[n],math.floor(pred)))

  df_temp = pd.DataFrame(output_messages, columns=['Nama Makanan', 'Rekomendasi Jenis Diskon', 'Jumlah Prediksi Terjual'])
  df_out = pd.concat([df_out, df_temp], ignore_index=True)
  
  return render_template('prediksi.html', periode=periode, data=df_out.to_html(justify='center'))
    
@app.route('/grafik', methods=['GET', 'POST'])
def grafik():
  return render_template ('grafik.html')

@app.route('/grafik/result', methods=['GET', 'POST'])
def garfikr():
  df = pd.read_excel('uploads/dataset.xlsx')
  data = df[['nama_item', 'week', 'jenis_diskon']]
  model = pickle.load(open('uploads/xgbr.model','rb'))

  label_encoders = {}
  cat_cols = ['nama_item', 'week', 'jenis_diskon']
  for col in cat_cols:
      label_encoders[col] = LabelEncoder()
      label_encoders[col].fit(data[col])
      data[col] = label_encoders[col].transform(data[col])

  sales_estimation = model.predict(data)
  rounded_sales = np.round(sales_estimation)
  result = np.concatenate((data, rounded_sales.reshape(-1, 1)), axis=1)
  result_df = pd.DataFrame(result, columns=['nama_item', 'week', 'jenis_diskon', 'predicted_value'])
  sum_by_week = result_df.groupby('week')['predicted_value'].sum()
  sum_by_week_df = sum_by_week.reset_index()
  sum_by_week_df.columns = ['week', 'predicted_value']
  sum_by_week_df

  month_value = request.form['month']
  month = month_weeks_dict()
  selected_weeks = month[month_value]

  selected_week_data_list = []

  for selected_week in selected_weeks:
      selected_week_data = result_df[result_df['week'] == selected_week]
      selected_week_data_list.append(selected_week_data)

  plt.figure(figsize=(10, 6))

  for selected_week_data in selected_week_data_list:
      plt.bar(selected_week_data['week'], selected_week_data['predicted_value'], label=f"Week {selected_week_data['week'].iloc[0]}")

  plt.xlabel('Minggu Ke-')
  plt.ylabel('Hasil Prediksi Penjualan')
  plt.title('Hasil Prediksi Penjualan Restoran Tuku Ramen')
  plt.grid(True)
  
  interactive_plot = mpld3.fig_to_html(plt.gcf())
  
  # Calculate and print sum of predicted_value for each selected week
  for selected_week_data in selected_week_data_list:
      week_sum = selected_week_data['predicted_value'].sum()
      print(f"Week {selected_week_data['week'].iloc[0]} - Sum Predicted Value: {week_sum}")

  return render_template('grafik.html', plot=interactive_plot, bulan=month_value)

if __name__ == "__main__":
  app.run(debug=True)