#!/bin/bash
echo "PyTest Wrapper Starting Bootstrap"

current_env=`conda info | grep "active environment :" | awk '{print $4}'`
if [ "${current_env}" == "traderl" ]; then
    pytest "$@"
else
    echo "TradeRL is not the current environment. Launch vscode from an activated environment."
    echo "Track this issue here: https://github.com/microsoft/vscode-python/issues/10668"
    echo "Using a native terminal: "
    echo "  conda activate traderl"
    echo "  code"
    echo 
fi