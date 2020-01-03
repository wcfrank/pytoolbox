set number
syntax on
set pastetoggle=<F2>
set hlsearch
"set tabs to have 4 spaces
set ts=4 
"set foldmethod=indent

let mapleader=','
inoremap <leader>w <Esc>:w<cr>

noremap <C-h> <C-w>h
noremap <C-j> <C-w>j
noremap <C-k> <C-w>k
noremap <C-l> <C-w>l

" Plugin
call plug#begin('~/.vim/plugged')
Plug 'mhinz/vim-startify'
Plug 'vim-airline/vim-airline'
Plug 'joshdick/onedark.vim'
"Plug 'Yggdroot/indentLine' 
Plug 'scrooloose/nerdtree' 
Plug 'kien/ctrlp.vim'
Plug 'easymotion/vim-easymotion'
Plug 'tpope/vim-surround'
Plug 'python-mode/python-mode', {'for': 'python', 'branch': 'develop' }
call plug#end()

colorscheme onedark
nnoremap <leader>v :NERDTreeFind<cr>
nnoremap <leader>g :NERDTreeToggle<cr>
nmap ss <Plug>(easymotion-s2)
" Plugin config
"let g:indentLine_enabled = 0
"let g:indentLine_setConceal = 0
let NERDTreeShowHidden = 1
let NERDTreeIgnore = ['__pycache__$', '\.pyc$', '\.DS_Store$', '\.egg-info$']
let g:ctrlp_map = '<c-p>'

let g:pymode_python = 'python'
let g:pymode_trim_whitespaces = 1
let g:pymode_doc = 1
let g:pymode_doc_bind = 'K'
let g:pymode_rope_goto_definition_bind = "<C-]>"
let g:pymode_lint = 1
let g:pymode_lint_checkers = ['pyflakes', 'pep8', 'mccabe', 'pylint']
let g:pymode_options_max_line_length = 120
