"""Run: python -m streamlit run streamlit_app.py"""
import hashlib
from pathlib import Path

import streamlit as st
from PIL import UnidentifiedImageError

from data_utils import HERE
from web_predict import decode_upload, load_model, run_prediction

st.set_page_config(page_title='Embryo Lab · Πρόβλεψη ζώνης',page_icon='🔬',layout='wide')
st.markdown('''<style>
.block-container{max-width:1250px;padding-top:2.5rem}
[data-testid="stMetric"]{background:#f0f5fa;border:1px solid #dce6ef;border-radius:12px;padding:16px}
div.stButton>button{border-radius:9px;min-height:46px}
</style>''',unsafe_allow_html=True)


@st.cache_resource(max_entries=1,show_spinner='Φόρτωση μοντέλου…')
def cached_model(path, modified_ns, byte_size, device, age):
    # File metadata invalidates cached weights after a checkpoint is replaced.
    return load_model(path,device,age)


st.title('Πρόβλεψη της περιοχής της ζώνης')
st.caption('Ανέβασε μια εικόνα εμβρύου, επίλεξε την ηλικία και δες τη μάσκα που προβλέπει το U-Net.')

with st.sidebar:
    st.header('Ρυθμίσεις πρόβλεψης')
    age=st.selectbox('Ηλικία εμβρύου',[25,69,117],format_func=lambda v:f'{v} ώρες')
    available=sorted((HERE/'runs').glob('*/best_model.pt'))
    choices=['Χειροκίνητη διαδρομή']+[str(p.parent.name) for p in available]
    selection=st.selectbox('Εκπαιδευμένο μοντέλο',choices)
    if selection=='Χειροκίνητη διαδρομή':
        checkpoint=st.text_input('Διαδρομή best_model.pt',value=f'runs/zona_age{age}_01/best_model.pt',key=f'checkpoint_{age}')
    else:
        checkpoint=str(HERE/'runs'/selection/'best_model.pt')
    device_label=st.radio('Εκτέλεση',['CPU','GPU (CUDA)'])
    device='cuda' if device_label=='GPU (CUDA)' else 'cpu'
    with st.expander('Όριο πρόβλεψης'):
        threshold=st.slider('Ελάχιστη πιθανότητα',min_value=.05,max_value=.95,value=.5,step=.05)
        st.caption('Αλλάζει τη μετατροπή της πιθανότητας σε μάσκα. Προεπιλογή: 0,50.')
    st.caption('Επίλεξε το μοντέλο που εκπαιδεύτηκε για την αντίστοιχη ηλικία.')

left,right=st.columns([1.4,1],gap='large')
with left:
    st.subheader('1. Εικόνα εμβρύου')
    upload=st.file_uploader('Επίλεξε PNG, JPG ή TIFF',type=['png','jpg','jpeg','tif','tiff'],key='image')
with right:
    st.subheader('2. Σωστή μάσκα — προαιρετικά')
    reference=st.file_uploader('Μάσκα PNG για σύγκριση',type=['png'],key='mask')
    st.caption('Μόνο για μετρικές: ένα κανάλι, τιμές 0/1 ή 0/255, ίδιες διαστάσεις με την εικόνα.')

image_bytes=upload.getvalue() if upload else None
mask_bytes=reference.getvalue() if reference else None
valid_image=False
if upload:
    try:
        preview=decode_upload(image_bytes)
        left.image(preview,caption=f'{upload.name} · {preview.width} × {preview.height}',use_container_width=True)
        valid_image=True
    except (ValueError,OSError,UnidentifiedImageError) as error:
        st.error(f'Δεν μπορεί να διαβαστεί η εικόνα: {error}')

path=Path(checkpoint).expanduser()
if not path.is_absolute():
    path=HERE/path
exists=path.is_file()
if not exists:
    st.info('Δεν βρέθηκε μοντέλο στη διαδρομή που επέλεξες. Δώσε το best_model.pt μιας ολοκληρωμένης εκπαίδευσης.')
stat=path.stat() if exists else None
identity=(age,str(path),stat.st_mtime_ns if stat else 0,stat.st_size if stat else 0,device,threshold,
          upload.name if upload else '',hashlib.sha256(image_bytes or b'').hexdigest(),
          hashlib.sha256(mask_bytes or b'').hexdigest())
if st.session_state.get('result_identity')!=identity:
    st.session_state.pop('prediction_result',None)

if st.button('Πρόβλεψη',type='primary',disabled=not(valid_image and exists),use_container_width=True):
    try:
        bundle=cached_model(str(path),stat.st_mtime_ns,stat.st_size,device,age)
        with st.spinner('Υπολογισμός της μάσκας…'):
            result=run_prediction(bundle,image_bytes,upload.name,age,mask_bytes,threshold,path.parent.name)
        st.session_state.prediction_result=result
        st.session_state.result_identity=identity
    except (ValueError,RuntimeError,OSError,KeyError,UnidentifiedImageError) as error:
        st.error(f'Η πρόβλεψη δεν ολοκληρώθηκε: {error}')
    except ImportError:
        st.error('Λείπει βιβλιοθήκη του project. Εγκατέστησε το requirements_web.txt στο ενεργό περιβάλλον.')

result=st.session_state.get('prediction_result')
if result:
    st.divider()
    st.subheader(f'Αποτέλεσμα · {age} ώρες')
    mask_tab,overlay_tab,comparison_tab=st.tabs(['Μάσκα','Πάνω στην εικόνα','Σύγκριση'])
    mask_tab.image(result['files']['prediction.png'],use_container_width=True)
    overlay_tab.image(result['files']['overlay.png'],use_container_width=True)
    comparison_tab.image(result['files']['comparison.png'],use_container_width=True)
    metrics=result['metrics']
    if metrics is not None:
        a,b,c=st.columns(3)
        a.metric('Dice',f'{metrics["dice"]:.2%}')
        b.metric('IoU / Jaccard',f'{metrics["iou"]:.2%}')
        c.metric('Pixel accuracy',f'{metrics["accuracy"]:.2%}')
        st.caption('Σύγκριση με τη μάσκα που ανέβασες, στην αρχική ανάλυση. Το accuracy περιλαμβάνει το φόντο.')
    else:
        st.caption('Για Dice, IoU και accuracy, ανέβασε και τη σωστή μάσκα της εικόνας.')
    a,b,c=st.columns(3)
    a.download_button('Λήψη μάσκας',result['files']['prediction.png'],file_name=f'zona_{age}h.png',mime='image/png')
    b.download_button('Λήψη overlay',result['files']['overlay.png'],file_name=f'overlay_{age}h.png',mime='image/png')
    c.download_button('Όλα τα αποτελέσματα ZIP',result['archive'],file_name=f'prediction_{age}h.zip',mime='application/zip')
