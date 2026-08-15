(cl:in-package #:statistical-learning.htm)


(defclass parameters ()
  ((%cells-per-column
    :initarg :cells-per-column
    :accessor cells-per-column
    :initform 20
    :documentation "Number of temporal memory cells per spatial pooler column. Each cell maintains its own sequence memory; higher values enable more complex temporal patterns but increase memory and computation cost.")
   (%activation-threshold
    :initarg :activation-threshold
    :accessor activation-threshold)
   (%learning-rate
    :initarg :learning-rate
    :accessor learning-rate)
   (%decay-rate
    :initarg :decay-rate
    :accessor decay-rate)
   (%drop-out-threshold
    :initarg :drop-out-threshold
    :accessor drop-out-threshold))
  (:documentation "Hierarchical Temporal Memory (HTM) configuration parameters, controlling only the TemporalMemory component. These parameters define cell behavior, connectivity, duty-cycle bounds, and learning dynamics essential for robust sequence learning and anomaly detection."))

(defclass htm-state ()
  ((%currently-active-neurons
    :initarg :currently-active-neurons
    :accessor currently-active-neurons)
   (%previously-active-neurons
    :initarg :previously-active-neurons
    :accessor previously-active-neurons)
   (%activity-hash-table
    :initarg :activity-hash-table
    :reader activity-hash-table)
   (%active-leafs
    :initarg :active-leafs
    :initform (make-hash-table)
    :reader active-leafs))
  (:default-initargs
   :previously-active-neurons (make-hash-table :test 'eq)
   :activity-hash-table (make-hash-table :test 'eq)
   :currently-active-neurons (make-hash-table :test 'eq)))

(defclass htm ()
  ((%ensemble
    :initarg :ensemble
    :initform (error "Ensemble required")
    :reader htm-ensemble
    :documentation "Wrapped ensemble (e.g., Random Forest)")
   (%parameters
    :initarg :parameters
    :reader htm-parameters
    :initform (error "Parameters required")
    :documentation "HTM-specific parameters")
   (%weights :initarg :weights
             :reader weights)
   (%state :initarg :state
           :reader htm-state
           :documentation "HTM state (active columns, temporal context)"))
  (:documentation "HTM wrapper for ensemble methods.")
  (:default-initargs
   :weights (make-hash-table :test 'eq)
   :state (make-instance 'htm-state)))

(defclass column-leaf ()
  ((%neurons :initarg :neurons :reader column-neurons))
  (:documentation "HTM column containing neurons with segments"))

(defclass neuron ()
  ()
  (:documentation "Neuron with synaptic segments for temporal pooling"))

(defun weights-neuron-weights (weights-hash-table neuron)
  "Return neuron-weights-hash-table for a neuron. Will create and insert an empty hash-table if it is not present."
  (ensure (gethash neuron weights-hash-table)
    (make-hash-table :test 'eq)))

(defun neuron-weights-leafs-weights (neuron-weights-hash-table column)
  "Returns leafs hash-table from neuron-weights hash-table. Will create and insert empty leafs hash-table into neuron-weights hash-table if it is absent, and then return empty hash-table."
  (ensure (gethash column neuron-weights-hash-table)
    (make-hash-table :test 'eq)))

(defun leafs-neuron-weight (leafs-hash-table neuron)
  "Get synaptic weight from the leafs-hash-table."
  (gethash neuron leafs-hash-table))

(defun (setf leafs-neuron-weight) (new-weight leafs-hash-table neuron)
  "Set synaptic weight in the leafs-hash-table."
  (setf (gethash neuron leafs-hash-table) new-weight))
