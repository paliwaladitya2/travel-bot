const { OpenAIEmbeddings } = require("langchain/embeddings/openai");
const { Milvus } = require("langchain/vectorstores/milvus");
const dotenv = require("dotenv");

dotenv.config();

exports.milvus = async function (options) {
    const option = this.parse(options);
    const text = option.text;
    const openai_api_key = option.key;


    const embeddings = new OpenAIEmbeddings({
        modalName: option.modalName,
        openAIApiKey: openai_api_key,
        batchSize: 512,
    });

    const vectorStore = await Milvus.fromExistingCollection(
        embeddings,
        {
            collectionName: option.text,
        }
    );

    const response = await vectorStore.similaritySearch(option.search, 2);
    return response;
}