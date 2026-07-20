import asyncio

from pypdf import PdfReader


async def pdf_text_extractor(filepath: str) -> str:
    """
    Function to extract text from a pdf by running the synchronous funciton
    in a separate thread asynchronously
    """
    return await asyncio.to_thread(_extract_pdf_text, filepath)


def _extract_pdf_text(filepath: str) -> str:
    pages = []
    pdf_reader = PdfReader(filepath, strict=True)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)
    content = "\n\n".join(pages)
    with open(filepath.replace("pdf", "txt"), "w", encoding="utf-8") as file:
        file.write(content)
    return content

# async def test():
#     print(await pdf_text_extractor("/Users/jvxoq/Documents/Projects/fastapi/talk-to-the-web/backend/uploads/Jason_Daniel_Product_Intern.pdf"))
# asyncio.run(test())
