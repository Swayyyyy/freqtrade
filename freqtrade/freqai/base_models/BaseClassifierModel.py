import logging
from time import time
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel
#导入准确率，AUC指标
from sklearn.preprocessing import label_binarize
from freqtrade.freqai.metrics import auc_score, accuracy_score_for_labels, accuracy_score_for_labels_with_weight, auc_score_for_multiclass

logger = logging.getLogger(__name__)

METRIC_MAP={
    "auc": auc_score,
    'auc_score_for_multiclass': auc_score_for_multiclass,
    'accuracy_for_labels': accuracy_score_for_labels,
    'accuracy_for_labels_with_weight': accuracy_score_for_labels_with_weight
}


class BaseClassifierModel(IFreqaiModel):
    """
    Base class for regression type models (e.g. Catboost, LightGBM, XGboost etc.).
    User *must* inherit from this class and set fit(). See example scripts
    such as prediction_models/CatboostClassifier.py for guidance.
    """

    def train(self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        Filter the training data and train a model to it. Train makes heavy use of the datakitchen
        for storing, saving, loading, and analyzing the data.
        :param unfiltered_df: Full dataframe for the current training period
        :param metadata: pair metadata from strategy.
        :return:
        :model: Trained model which can be used to inference (self.predict)
        """

        logger.info(f"-------------------- Starting training {pair} --------------------")

        start_time = time()

        # filter the features requested by user in the configuration file and elegantly handle NaNs
        features_filtered, labels_filtered = dk.filter_features(
            unfiltered_df,
            dk.training_features_list,
            dk.label_list,
            training_filter=True,
        )

        start_date = unfiltered_df["date"].iloc[0].strftime("%Y-%m-%d")
        end_date = unfiltered_df["date"].iloc[-1].strftime("%Y-%m-%d")
        logger.info(
            f"-------------------- Training on data from {start_date} to "
            f"{end_date} --------------------"
        )
        # split data into train/test data.
        dd = dk.make_train_test_datasets(features_filtered, labels_filtered)
        if not self.freqai_info.get("fit_live_predictions_candles", 0) or not self.live:
            dk.fit_labels()
        dk.feature_pipeline = self.define_data_pipeline(threads=dk.thread_count)

        (dd["train_features"], dd["train_labels"], dd["train_weights"]) = (
            dk.feature_pipeline.fit_transform(
                dd["train_features"], dd["train_labels"], dd["train_weights"]
            )
        )

        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) != 0:
            (dd["test_features"], dd["test_labels"], dd["test_weights"]) = (
                dk.feature_pipeline.transform(
                    dd["test_features"], dd["test_labels"], dd["test_weights"]
                )
            )

        logger.info(
            f"Training model on {len(dk.data_dictionary['train_features'].columns)} features"
        )
        logger.info(f"Training model on {len(dd['train_features'])} data points")

        model = self.fit(dd, dk)

        end_time = time()
        if self.freqai_info.get("filter_model", {}) and self.freqai_info["filter_model"].get("enabled", False):
            metric = self.freqai_info["filter_model"].get("metric", "accuracy")
            metric_func = METRIC_MAP[metric]
            if self.freqai_info["filter_model"].get("input_type") == 'prob':
                metric_value = metric_func(label_binarize(dd["test_labels"], classes=self.class_names), model.predict_proba(dd["test_features"]), **self.freqai_info["filter_model"].get("metric_kwargs", {}))
            else:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                le.fit_transform(self.class_names)
                metric_value = metric_func(dd["test_labels"].values.reshape(-1), model.predict(dd["test_features"]), le = le, **self.freqai_info["filter_model"].get("metric_kwargs", {}))
            metric_threshold = self.freqai_info["filter_model"].get("threshold", 0)
            if metric_value < metric_threshold:
                self.model = None
                logger.info(f'Model for pair {pair}, {metric} value is {metric_value}, below threshold {metric_threshold}, remove model')
            else:
                logger.info(f'Model for pair {pair}, {metric} value is {metric_value}, above threshold {metric_threshold}, keeping model')
        logger.info(
            f"-------------------- Done training {pair} "
            f"({end_time - start_time:.2f} secs) --------------------"
        )

        return model

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> tuple[DataFrame, npt.NDArray[np.int_]]:
        """
        Filter the prediction features data and predict with it.
        :param unfiltered_df: Full dataframe for the current backtest period.
        :return:
        :pred_df: dataframe containing the predictions
        :do_predict: np.array of 1s and 0s to indicate places where freqai needed to remove
        data (NaNs) or felt uncertain about data (PCA and DI index)
        """
        dk.find_features(unfiltered_df)
        filtered_df, _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )

        dk.data_dictionary["prediction_features"] = filtered_df

        dk.data_dictionary["prediction_features"], outliers, _ = dk.feature_pipeline.transform(
            dk.data_dictionary["prediction_features"], outlier_check=True
        )

        predictions = self.model.predict(dk.data_dictionary["prediction_features"])
        if self.CONV_WIDTH == 1:
            predictions = np.reshape(predictions, (-1, len(dk.label_list)))

        pred_df = DataFrame(predictions, columns=dk.label_list)

        predictions_prob = self.model.predict_proba(dk.data_dictionary["prediction_features"])
        if self.CONV_WIDTH == 1:
            predictions_prob = np.reshape(predictions_prob, (-1, len(self.model.classes_)))
        pred_df_prob = DataFrame(predictions_prob, columns=self.model.classes_)

        pred_df = pd.concat([pred_df, pred_df_prob], axis=1)

        if dk.feature_pipeline["di"]:
            dk.DI_values = dk.feature_pipeline["di"].di_values
        else:
            dk.DI_values = np.zeros(outliers.shape[0])
        dk.do_predict = outliers

        return (pred_df, dk.do_predict)
