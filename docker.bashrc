echo -e "\033[38;5;82m\033[1m"
echo " __      __  _____          _____        _     "
echo " \ \    / / |  __ \   /\   |  __ \ /\   | |    "
echo "  \ \  / /__| |__) | /  \  | |__) /  \  | | __ "
echo "   \ \/ / _ \  _  / / /\ \ |  ___/ /\ \ | |/ / "
echo "    \  /  __/ | \ \/ ____ \| |  / ____ \|   <  "
echo "     \/ \___|_|  \__/    \___| /_/    \___|\_\ "
echo -e "\033[22m"
echo "Alias: verapak == python /src/VERAPAK"
echo "       Call 'verapak' for usage"

alias verapak='python /src/VERAPAK'

$(cd /src/VERAPAK && git fetch > /dev/null 2>&1)
verapak_behind_count=$(cd /src/VERAPAK && git rev-list --count HEAD..@{u})

if [ "$verapak_behind_count" -gt "0" ]; then
    echo ""
    echo "Your version of VeRAPAk is $verapak_behind_count commit(s) behind!"
    echo "    Update to the most recent version using 'docker pull yodarocks1/verapak:latest'"
fi

echo -e "\033[39m"


# If not running interactively, don't do anything
[ -z "$PS1" ] && return

PS1="\[\e[38;5;82m\e[1m\]verapak-docker\[\e[m\] \[\e[33m\]\w\[\e[m\] > "

# don't put duplicate lines in the history. See bash(1) for more options
# ... or force ignoredups and ignorespace
HISTCONTROL=ignoredups:ignorespace

# append to the history file, don't overwrite it
shopt -s histappend

# for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
HISTSIZE=1000
HISTFILESIZE=2000

# check the window size after each command and, if necessary,
# update the values of LINES and COLUMNS.
shopt -s checkwinsize

# make less more friendly for non-text input files, see lesspipe(1)
[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

# set variable identifying the chroot you work in (used in the prompt below)
if [ -z "$debian_chroot" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

# If this is an xterm set the title to user@host:dir
case "$TERM" in
xterm*|rxvt*)
    PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
    ;;
*)
    ;;
esac

# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# Alias definitions.
# You may want to put all your additions into a separate file like
# ~/.bash_aliases, instead of adding them here directly.
# See /usr/share/doc/bash-doc/examples in the bash-doc package.

if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
#if [ -f /etc/bash_completion ] && ! shopt -oq posix; then
#    . /etc/bash_completion
#fi
