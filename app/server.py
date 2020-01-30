import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.requests import Request

from cleaning import clean

export_file_url = 'https://drive.google.com/uc?export=download&id=1C5s0l2jOZAUGnUEsq7k3TF-SlyL9K7dg'
export_file_name = 'export.pkl'

classes = ['lidl', 'spisestuerne', 'netto','seveneleven']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'FileUpload.html'
    return HTMLResponse(html_file.open().read())

@app.route('/dostuff')
async def dostuff(request):
    prediction = learn.predict('lidl')
    return JSONResponse({'result': str(prediction)})

@app.route('/upload', methods = ['POST'])
async def upload_file(request):

    bank='danskebank'
    form = await request.form()
    filename=form['file'].filename
    file_bytes=form['file'].file.read()
    file = open(BytesIO(file_bytes))
    #file=form['file']

    # read the large csv file with specified chunksize 
    df_chunk = pd.read_csv(file,delimiter=";", encoding='cp1252', chunksize=200)
    # append each chunk df here 
    chunk_list = []

    # Each chunk is in df format
    for chunk in df_chunk:  
        # append the chunk to list
        chunk_list.append(chunk)
        
    # concat the list into dataframe 
    df= pd.concat(chunk_list)
    # Filter out unimportant columns
    df = df[['Dato','Beløb','Tekst']]
    df.rename(columns={'Dato':'Date','Beløb':'Amount','Tekst':'Text'}, inplace=True)
    df['Amount'] = df['Amount'].replace({'\.':''}, regex = True)
    df['Amount'] = df['Amount'].replace({'\,':'.'}, regex = True)
    
    #Change Datatype of amount to float
    df['Amount'] = df['Amount'].astype('float64', copy=False)
    #Delete all positive transfers
    df=df[df.Amount <0.0]
                        
    #Delete weird brackets (((
    df['Text'] = df['Text'].replace({'\)\)\)\)':''}, regex = True)
    df['Text'] = df['Text'].replace({'\)\)\)':''}, regex = True)

    return("Bin durch")
    """
    print("I am in upload method")
    bank = request.form.get('bankselect')
    f = request.files['file']
    f.save(f.filename)
    print("i now call to clean")
    result = clean(f.filename,bank,learn)
    return result
    """

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict('lidl')
    prediction = learn.predict('lidl')
    prediction = learn.predict('lidl')
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
