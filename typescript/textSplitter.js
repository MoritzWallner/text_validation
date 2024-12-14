function splitTextWithPosition(article) {
  // Split the article by lines first
  const lines = article.split("\n");
  const sections = [];
  let currentSection = { heading: "", content: "" };

  lines.forEach((line) => {
    // Check if the line is a heading (starts with #, ##, ###, etc.)
    if (/^#{1,6}\s/.test(line)) {
      // When a heading is found, push the current section to sections if it exists
      console.log(currentSection);
      sections.push(currentSection);
      // Start a new section with the current heading
      currentSection = { heading: line, content: "" };
    } else {
      // Add the line to the content of the current section
      currentSection.content += line + "\n";
    }
  });

  // Push the last section if it exists
  if (currentSection.heading || currentSection.content) {
    sections.push(currentSection);
  }

  return sections;
}

function removeMarkdownFromText(text) {
  // Remove image links (e.g., ![alt text](image_url))
  let plainText = text.replace(/!\[.*?\]\(.*?\)/g, "");

  // Remove Markdown headers and formatting symbols (e.g., ### for headers, ** for bold, * for italics)
  plainText = plainText.replace(/[#*]+/g, "");

  // Remove hyperlinks but keep the link text (e.g., [link text](url) -> link text)
  plainText = plainText.replace(/\[(.*?)\]\(.*?\)/g, "$1");

  return plainText;
}

const article = `# Overview
At JUHUU, our locking systems offer maximum flexibility and security for a wide range of infrastructure, from bike boxes to locker systems. The locking system consists of two core components: Lock Control Units and Locks, designed to work seamlessly with our software and IoT solutions. 

## Locking System Components 

**1. Lock Control Units:** 
Our lock control units (LCUs) manage the locking and unlocking of devices, transmitting signals from the [**IoT Module**]() to the connected lock system. We offer three main types: 
- **SCU (Single Control Unit)**: This unit controls one lock and can be connected in series with up to 40 SCUs for distributed setups. This solution is recommended when there's a larger distance between locks (e.g., multiple bike boxes in a row). 
- **12CU**: A control unit that supports up to 12 locks. This is suitable for setups where locks are closer together, such as luggage lockers. 
- **48CU**: Capable of controlling up to 48 locks. Ideal for compact setups, such as package stations, where the locks are closely grouped. 

In situations where more flexibility is needed, such as distance between locks, the SCU is ideal but comes with higher costs due to individual units for each lock. For compact arrangements, 12CU and 48CU provide efficient solutions. 

### 2. Locks:
Our locking systems currently support the following models: 

- [**KR-S98A**](https://docs.juhuu.app/articles/6707e36bf1f29963bc4c7fe5)
- [**KR-S98B**](https://docs.juhuu.app/articles/6707e38bf1f29963bc4c800f)
- [**KR-S98S**](https://docs.juhuu.app/articles/6707e77bf1f29963bc4c88a6)
- [**KR-S79N**](https://docs.juhuu.app/articles/6707da9bf1f29963bc4c6e10)
- [**KR-S70N (recommended)**](https://docs.juhuu.app/articles/6707d5daf1f29963bc4c5f99)
- [**KR-S97**](https://docs.juhuu.app/articles/6707dd52f1f29963bc4c7742)
- [**KR-S98C**](https://docs.juhuu.app/articles/6707e6d5f1f29963bc4c83b5) 

All locks in this list are fully compatible with our system and offer different specifications to fit various use cases, including waterproofing and different holding forces. 

**3. Hooks:** 
In addition to locks, we also provide compatible hooks, allowing for further customization depending on the use case. ( Here, an image will be provided.) 

## Custom Lock Solutions 

If a customer requires non-standard or custom locking mechanisms (e.g., electric cylinders, specific motors), our **Customized Lock Solution** enables integration of these into the JUHUU system. By using our **IoT Module**, we can adjust the system to work with a variety of third-party locking solutions, giving maximum flexibility. For this, the customer can contact us for bespoke solutions and adjustments. 

## See Also: 

- [**SCU**](https://docs.juhuu.app/articles/670500951113f7de8a8690c9): Learn more about the Single Control Unit and its capabilities. 
- [**12CU**](https://docs.juhuu.app/articles/670500951113f7de8a8690c9): Explore how the 12-lock control unit fits into compact systems. 
- [**48CU**](https://docs.juhuu.app/articles/670500951113f7de8a8690c9): Learn about the 48-lock control unit for larger setups. 
- [**Customized Lock Solution**](https://docs.juhuu.app/articles/6705012f1113f7de8a8690f1): Discover how we can adapt to your unique locking need
`;

// const plainText = removeMarkdownFromText(article);

// console.log(article);
// console.log(plainText);

const textPartArray = splitTextWithPosition(article);

console.log(JSON.stringify(textPartArray, null, 2));
