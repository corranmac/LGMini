#@title Stretchsize processing

!rm -rf stretchsize
!mkdir stretchsize

original_file = ''
st_width = 256
st_height = 256
if do_stretchsize:
  new_input_files = []
  #@markdown `do_stretchsize` disables `do_resize`
  do_resize = False
  #@markdown `do_stretchsize` always does `skip_gt`
  skip_gt = True

  for image_path in input_files:
    __, image_name = os.path.split(image_path)
    im = Image.open(image_path)
    st_width, st_height = im.size
    if st_width > st_height:
      im1 = im.resize((st_width, st_width))
    else:
      im1 = im.resize((st_height, st_height))
    stretched_path = os.path.join("stretchsize", image_name)
    im1.save(stretched_path)
    new_input_files.append(stretched_path)
  print("Input files:", new_input_files)
  input_files = new_input_files
  try:
    im = Image.open(ss_size_parent)
    st_width, st_height = im.size
  except:
    st_width, st_height = 256
