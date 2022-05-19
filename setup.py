#Swinir Stuff
!git clone -qq https://github.com/Lin-Sinorodin/SwinIR_wrapper.git
try:
  from SwinIR_wrapper.SwinIR_wrapper import SwinIR_SR
except:
  try:
    import collections.abc as container_abcs
    from SwinIR_wrapper.SwinIR_wrapper import SwinIR_SR
  except:
    pass


#Setup stuff
ram_gb = round(virtual_memory().total / 1024**3, 1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

%matplotlib inline
%config InlineBackend.figure_format = 'svg'
%config InlineBackend.rc = {'figure.figsize': (10.0, 10.0)}


#!git checkout better-caching
!git checkout develop
!pip install -e .


!rm -rf output
!mkdir output
output.clear()


# Import model
model = get_rudalle_model('Malevich', pretrained=True, fp16=True, device=device)
vae = get_vae().to('cuda')
tokenizer = get_tokenizer()

mount_drive = False #@param {type:"boolean"}
mount_location = 'drive' #@param {type:"string"}
mount_location = '/content/' + mount_location


save_checkpoint_in_drive = False #@param {type:"boolean"}  
restore_checkpoint_from_drive = False #@param {type:"boolean"} 
drive_checkpoint_filepath = '/MyDrive/lookingglass_dalle_last.pt' #@param {type:"string"} 

save_output_in_drive = False #@param {type:"boolean"}
drive_output_filepath = '/MyDrive/LookingGlassOutput' #@param {type:"string"}

drive_output_filepath = mount_location + drive_output_filepath

if mount_drive and not os.path.exists(mount_location):
  drive.mount(mount_location)
  Path(drive_output_filepath).mkdir(parents=True, exist_ok=True)
if mount_drive == False:
  save_checkpoint_in_drive = False
  restore_checkpoint_from_drive = False
  save_output_in_drive = False

 file_selector_glob = "images/*"  # @param {type:"string"}
input_files = glob.glob(file_selector_glob, recursive=True)
for i in input_files:
  if "_" in i:
    raise ValueError("Please remove all underscores (the _ character) from your files before proceeding!")
  if "'" in i:
    raise ValueError("Please remove all apostrophes (the ' character) from your files before proceeding!")


print("Input files:", input_files)
if len(input_files) == 0:
  print("Your input files are empty! This will error out - make sure your file_selector_glob is formatted correctly!")


#@markdown # Finetuning Parameters
#@markdown The amount of epochs that training occurs for. Turn down if the images are too similar to the base image. Turn up if they're too different. Use this for fine adjustments.
epoch_amt =   50# @param {type:"number"}
#@markdown Universe similarity determines how close to the original images you will receive. Higher similarity produces images that try to stick closely to the original. Lower similarity produces images that use the original more as inspiration. **If you are getting spooky Russian AI ghosts, try turning your similarity higher or training for longer.** 
universe_similarity = "Low"  # @param ["Ultra-High","High", "Medium", "Low","Ultra-Low"]

#@markdown Advanced users can manually set learning rate (the parameter that universe similarity controls). If you don't know what "learning rate" means, leave this at -1, which disables manual input.
learning_rate =  -1# @param {type:"number"}
if learning_rate == -1:
  if universe_similarity == "High":
      learning_rate = 1e-4
  elif universe_similarity == "Medium":
      learning_rate = 2e-5
  elif universe_similarity == "Low":
      learning_rate = 1e-5
  elif universe_similarity == "Ultra-Low":
      learning_rate = 1e-6
  elif universe_similarity == "Ultra-High":
      learning_rate = 2e-4
  else:
      learning_rate = 1e-5
      
#@markdown Additionally, advanced users can manually set the weight decay of the fine-tuning system. This has complicated outcomes that are not easily explainable, but, generally, it is how much of the "original model" is retained. Leave as -1 to use the default system values.
weight_decay = -1 # @param {type:"number"}
#@markdown Enabling the DWT module causes Looking Glass to output images at doubled resolution (512x512) at the expense of quality.
dwt=True # @param {type:"boolean"}
#@markdown <br></br>
#@markdown # Captioning Parameters
#@markdown Input text can influence the end results, so you have the option to change it. **Input text is automatically translated from any language to Russian.**
input_text = ""  # @param {type:"string"}

if input_text != "":
  if len(input_text) < 10:
    raise ValueError("Your input text is too short. Please make it longer!")
  if len(input_text) > 100:
    raise ValueError("Your input text is too long. Please make it shorter!")
    
if input_text == "":
  input_text = "\u0420\u0438\u0447\u0430\u0440\u0434 \u0414. \u0414\u0436\u0435\u0439\u043C\u0441"
else:
  input_lang = ts.language(input_text).result.alpha2
  if input_lang != 'ru':
    if deepl_api_key != "":
      input_text = DeeplTranslator(api_key=deepl_api_key, source=input_lang, target='ru', use_free_api=True).translate(input_text) 
    else:
      input_text = ts.translate(input_text, "ru").result

#@markdown Enabling `use_filename` will cause your input text to be overwritten by the filenames of your pictures, translated into Russian. For example, a file named "picture of a boy.png" will be captioned "картина мальчика".
use_filename = False  #@param {type: "boolean"}


class Args():
    def __init__(self):
        self.text_seq_length = model.get_param('text_seq_length')
        self.total_seq_length = model.get_param('total_seq_length')
        self.epochs = epoch_amt
        self.save_dir = 'checkpoints'
        self.model_name = 'lookingglass'
        self.save_every = 2000
        self.prefix_length = 10
        self.bs = 1
        self.clip = 0.24
        self.lr = learning_rate
        self.warmup_steps = 50
        self.wandb = False

torch_args = Args()
if not os.path.exists(torch_args.save_dir):
    os.makedirs(torch_args.save_dir)

#@markdown #Output Resizing
#@markdown If you'd like to change the shape or size of the output from its default 256x256 set "resize" to true.<br>Note that this is **much slower**.
do_resize = False  # @param {type:"boolean"}
#if do_resize:
#    low_mem = True
width =   512# @param {type:"number"}
height =   240# @param {type:"number"}
token_width = round(width / 8)
token_height = round(height / 8)

#@markdown <br><br>
#@markdown #Stretchsizing
#@markdown A more crude form of image resizing that squishes your initial image down to 256x256, and then expands the output images back to your original image's aspect ratio. May result in artifacts, but runs much faster than Output Resizing.
#@markdown <br>CURRENTLY INCOMPATIBLE WITH OUTPUT RESIZING. I WILL FIX THIS EVENTUALLY I'M JUST LAZY.
do_stretchsize = False  # @param {type:"boolean"}

ss_size_parent = input_files[0]
if do_stretchsize:
    ss_realesrgan = get_realesrgan("x2", device=device)

original_folder = re.sub(r'[/*?]', '-', file_selector_glob)
print("Identifier", original_folder)
