(cl:in-package #:cl-grf.tree-protocol)


(defun needs-split-p (training-state leaf)
  (needs-split-p-with-mode (needs-split-p-mode training-state)
                           training-state
                           leaf))


(defun split (training-state leaf)
  (split-with-mode (split-mode training-state)
                   training-state
                   leaf))


(defun clone-training-state (training-state new-data)
  (check-type training-state fundamental-training-state)
  (cl-ds.utils:quasi-clone training-state
                           :training-data new-data))


(defun force-tree (node)
  (force-tree* (lparallel:force node)))
