using Microsoft.AspNetCore.Mvc;
using IndirectTax.Data;
using IndirectTax.Models;
using System.Collections.Generic;

namespace IndirectTax.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class ReconcillationSyntheticController : ControllerBase
    {
        private readonly ReconcillationSyntheticRepository _repository;

        public ReconcillationSyntheticController(ReconcillationSyntheticRepository repository)
        {
            _repository = repository;
        }

        [HttpPost]
        public ActionResult<IEnumerable<ReconcillationSynthetic>> Post([FromBody] IEnumerable<ReconcillationSynthetic> items)
        {
            var data = _repository.BulkAddAndGetAll(items);
            return Ok(data);
        }
        [HttpPost]
        public ActionResult Update([FromBody] ReconcillationUpdate items)
        {
             _repository.Update(items);
            return Ok();
        }

        [HttpGet]
        public ActionResult<IEnumerable<ReconcillationSynthetic>> Get()
        {
            var data = _repository.GetAll();
            return Ok(data);
        }
    }
}
