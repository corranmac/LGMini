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
