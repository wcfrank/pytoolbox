set number
syntax on
set pastetoggle=<F2>
set hlsearch
set foldmethod=indent


let mapleader=','
inoremap <leader>w <Esc>:w<cr>


noremap <C-h> <C-w>h
noremap <C-j> <C-w>j
noremap <C-k> <C-w>k
noremap <C-l> <C-w>l


call plug#begin('~/.vim/plugged')

Plug 'mhinz/vim-startify'
Plug 'vim-airline/vim-airline'
"Plug 'vim-airline/vim-airline-themes'
"Plug 'arcticicestudio/nord-vim'
"Plug 'Yggdroot/indentLine' 
"Plug 'itchyny/lightline.vim'
Plug 'joshdick/onedark.vim'
"Plug 'KeitaNakamura/neodark.vimm'

call plug#end()
colorscheme onedark 
