from django.test import TestCase

from apps.ml.intervention_classifier.SVM import SVM


class MLTests(TestCase):
    def test_svm_algorithm(self):
        input_data = {"message": "Typhoon"}

        response = SVM().predict(input_data)
        print(response)
        # self.assertEqual('OK', response['status'])
        # self.assertTrue('label' in response)
        # self.assertEqual('<=50K', response['label'])
