import src
import os


def main():

	train = src.DataProcessing(os.path.join('data', 'raw', 'trainSet.csv'), os.path.join('data', 'raw', 'train'))
	embedding_dict = src.DataProcessing.load_embedding(os.path.join('data', 'interim', 'cc.en.300.vec'))
	train.read_file()
	train.augment_text()
	train.shuffle_data()
	train.convert_to_vector(embedding_dict)
	train.save_data()

	test = src.DataProcessing(os.path.join('data', 'raw', 'candidateTestSet.txt'), os.path.join('data', 'raw', 'test'))
	test.read_file()
	test.convert_to_vector(embedding_dict)
	test.save_data()

	clf = src.Model(os.path.join('data', 'raw', 'train'))
	clf.create_model(random_state=1, n_jobs=-1)
	clf.train()
	clf.predict(os.path.join('content', 'test'))
	clf.save_predictions()



if __name__ == "__main__":
    main()