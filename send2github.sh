#!/bin/bash

read -p "Comments: " comment
git add  .
git commit -m "$comment"
git push

